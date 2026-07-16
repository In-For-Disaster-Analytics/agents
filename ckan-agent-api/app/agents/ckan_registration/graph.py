from __future__ import annotations

import sqlite3
import uuid
from functools import lru_cache
from typing import Any

from langchain_core.tracers.context import collect_runs
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.ckan_registration.logging_config import log_graph_end, log_graph_start, logger
from app.evals.online_evaluators import post_feedback_async
from app.agents.ckan_registration.nodes import (
    make_approval_node,
    make_file_metadata_node,
    make_geo_apply_node,
    make_geo_approval_node,
    make_intake_node,
    make_legacy_command_node,
    make_revise_field_node,
    make_safe_apply_node,
    make_show_node,
    route_from_intake,
)
from app.agents.ckan_registration.persona_nodes import (
    make_clarify_node,
    make_persona_node,
    make_propose_node,
    make_schema_select_node,
    route_after_persona,
    route_after_propose,
)
from app.agents.ckan_registration.schemas import AgentRunResponse, CkanResumeRequest, CkanRunRequest
from app.agents.ckan_registration.state import CkanRegistrationState
from app.settings import Settings, get_settings


def build_graph(settings: Settings, checkpointer=None):
    builder = StateGraph(CkanRegistrationState)
    builder.add_node("intake", make_intake_node(settings))
    builder.add_node("dry-run", make_legacy_command_node(settings, "dry-run"))
    builder.add_node("approval", make_approval_node())
    builder.add_node("apply", make_safe_apply_node(settings))
    builder.add_node("show", make_show_node(settings))
    # Gated geo transform path: persona proposes → human approves → run on Abaco.
    builder.add_node("geo-approval", make_geo_approval_node())
    builder.add_node("geo-apply", make_geo_apply_node(settings))
    builder.add_node("revise-field", make_revise_field_node(settings))

    # `analyze`/`revise` route to "metadata" from route_from_intake. Behind the
    # CKAN_PERSONA_CHAT flag those actions run the persona-chat subgraph (author →
    # clarify(interrupt) → propose); otherwise they use the single-pass metadata node.
    if settings.persona_chat_enabled:
        metadata_target = "schema_select"
        builder.add_node("schema_select", make_schema_select_node(settings))
        builder.add_node("persona", make_persona_node(settings))
        builder.add_node("clarify", make_clarify_node())
        builder.add_node("propose", make_propose_node(settings))
        builder.add_edge("schema_select", "persona")
        builder.add_conditional_edges(
            "persona", route_after_persona, {"clarify": "clarify", "propose": "propose"}
        )
        builder.add_edge("clarify", "persona")
        # A persona may propose a geo transform → route to the gated geo-approval node.
        builder.add_conditional_edges("propose", route_after_propose, {"geo-approval": "geo-approval", "END": END})
    else:
        metadata_target = "metadata"
        builder.add_node("metadata", make_file_metadata_node(settings))
        builder.add_edge("metadata", END)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges(
        "intake",
        route_from_intake,
        {
            "metadata": metadata_target,
            "dry-run": "dry-run",
            "apply": "approval",
            "geo-transform": "geo-approval",
            "show": "show",
            "revise-field": "revise-field",
        },
    )
    builder.add_edge("approval", "apply")
    builder.add_edge("geo-approval", "geo-apply")
    builder.add_edge("dry-run", END)
    builder.add_edge("apply", END)
    builder.add_edge("geo-apply", END)
    builder.add_edge("show", END)
    builder.add_edge("revise-field", END)
    return builder.compile(checkpointer=checkpointer)


def make_checkpointer(settings: Settings):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
        return SqliteSaver(connection)
    except Exception:
        return InMemorySaver()


def config_for_thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


class CkanRegistrationRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.graph = build_graph(settings, checkpointer=make_checkpointer(settings))

    def invoke(self, request: CkanRunRequest) -> AgentRunResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        thread_id = str(payload.get("session_id") or uuid.uuid4().hex)
        payload["session_id"] = thread_id
        state: CkanRegistrationState = {
            "thread_id": thread_id,
            "action": str(payload.get("action") or ""),
            "request": payload,
        }
        # Log graph invocation
        has_data = bool(
            payload.get("files")
            or payload.get("uploaded_files")
            or payload.get("inline_files")
            or payload.get("upload_dir")
            or payload.get("upload_dirs")
            or payload.get("source_url")
            or payload.get("source_urls")
        )
        log_graph_start(
            thread_id,
            state["action"],
            {"has_session": bool(payload.get("session_id")), "has_data": has_data},
        )
        try:
            with collect_runs() as cb:
                result = self.graph.invoke(state, config=config_for_thread(thread_id))
            log_graph_end(thread_id, result.get("status", "unknown"), error=result.get("error"))
            if cb.traced_runs:
                post_feedback_async(cb.traced_runs[0].id, result)
            return self._response(thread_id, result)
        except Exception as e:
            log_graph_end(thread_id, "error", error=str(e))
            raise

    def resume(self, thread_id: str, request: CkanResumeRequest) -> AgentRunResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        payload["session_id"] = thread_id
        logger.info(f"📋 RESUMING workflow: Thread={thread_id}")
        try:
            with collect_runs() as cb:
                result = self.graph.invoke(Command(resume=payload), config=config_for_thread(thread_id))
            log_graph_end(thread_id, result.get("status", "unknown"), error=result.get("error"))
            if cb.traced_runs:
                post_feedback_async(cb.traced_runs[0].id, result)
            return self._response(thread_id, result)
        except Exception as e:
            log_graph_end(thread_id, "error", error=str(e))
            raise

    def pending_interrupt(self, thread_id: str) -> bool:
        """True if the thread is paused mid-run awaiting a resume (e.g. a clarification
        interrupt). Used by the chat-compat endpoint to route a follow-up to resume()."""
        try:
            snapshot = self.graph.get_state(config_for_thread(thread_id))
        except Exception:
            return False
        return bool(getattr(snapshot, "next", None))

    def show(self, thread_id: str, request_headers: dict[str, str] | None = None) -> AgentRunResponse:
        request = CkanRunRequest(action="show", session_id=thread_id, request_headers=request_headers)
        return self.invoke(request)

    @staticmethod
    def _response(thread_id: str, state: dict[str, Any]) -> AgentRunResponse:
        interrupts = state.get("__interrupt__")
        requires_action = None
        if interrupts:
            first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
            requires_action = getattr(first, "value", first)
        result = state.get("result") or {}
        return AgentRunResponse(
            ok=not bool(state.get("error")) and bool(result.get("ok", True)),
            thread_id=thread_id,
            command=result.get("command") or state.get("action"),
            status=state.get("status"),
            result=result,
            requires_action=requires_action,
            error=state.get("error") or result.get("error"),
        )


@lru_cache
def get_runner() -> CkanRegistrationRunner:
    return CkanRegistrationRunner(get_settings())


# Module-level export for LangGraph Studio / langgraph dev — no custom checkpointer,
# Studio manages its own persistence.
graph = build_graph(get_settings())
