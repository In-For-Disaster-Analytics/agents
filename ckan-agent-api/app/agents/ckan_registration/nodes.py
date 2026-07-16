from __future__ import annotations

import csv
import difflib
import hashlib
import json
import mimetypes
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langgraph.types import interrupt

from app.agents.ckan_registration.ckan_client import CkanClient
from app.agents.ckan_registration.legacy_worker import LegacyCkanWorker, state_thread_id
from app.agents.ckan_registration.logging_config import (
    log_error,
    log_interrupt,
    log_node_entry,
    log_node_exit,
    log_routing_decision,
    logger,
)
from app.agents.ckan_registration.state import CkanRegistrationState
from app.prompts import PromptRegistry, get_prompt_registry
from app.settings import Settings

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


APPLY_APPROVAL = "REGISTER"
DELETE_STALE_APPROVAL = "DELETE_STALE_RESOURCES"


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _langchain_messages(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    role_map = {"assistant": "ai", "system": "system", "user": "human"}
    return [(role_map.get(message["role"], message["role"]), message["content"]) for message in messages]


def _invoke_openai_chat(
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    # Delegates to the shared helper in app.llm (single source for the chat call).
    from app import llm

    return llm.invoke_chat(
        messages,
        model=settings.ckan_llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or "",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def normalize_action(value: object | None) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "dryrun": "dry-run",
        "dry-run": "dry-run",
        "dry run": "dry-run",
        "validate": "dry-run",
        "preview": "dry-run",
        "status": "show",
        "inspect": "show",
        "register": "apply",
        "transform": "geo-transform",
        "geo-transform": "geo-transform",
        "transform-status": "transform-status",
        "revise_field": "revise-field",
        "revise-field": "revise-field",
    }
    return aliases.get(text, text)


def infer_action(request: dict[str, Any]) -> str:
    """Infer action from explicit request fields only (no regex fallback).
    
    If action is not explicitly provided, returns "analyze" as a safe default.
    LLM-based refinement happens in the planning node.
    """
    explicit = normalize_action(request.get("action") or request.get("command"))
    if explicit in {"analyze", "revise", "dry-run", "apply", "show"}:
        return explicit
    message = str(request.get("message") or "").strip()
    message_lower = message.lower()
    if request.get("approval") or message.upper() == APPLY_APPROVAL:
        return "apply"
    if re.search(r"\bregister\b", message_lower):
        return "apply"
    if re.search(r"\b(dry[- ]?run|validate|validation|preview|compare|diff)\b", message_lower):
        return "dry-run"
    if re.search(r"\b(show|status|state|debug)\b", message_lower):
        return "show"
    if request.get("exclude_resources"):
        return "revise" if request.get("session_id") else "analyze"
    if request.get("dataset"):
        return "analyze"
    if request.get("session_id") and re.search(r"\b(update|updating|existing)\b", message_lower):
        return "dry-run"
    return "analyze"


def llm_classify_action(
    settings: Settings,
    user_message: str,
    has_session_id: bool,
    has_data_input: bool,
) -> str:
    """Use LLM to classify user intent into an action: analyze, revise, dry-run, apply, or show."""
    if not settings.openai_api_key or not user_message:
        return "analyze"  # Safe default if LLM unavailable

    try:
        # Load action classification prompt from registry
        registry = get_prompt_registry()
        prompt_template = registry.load("ckan_registration", "action_classify")
        system_prompt = prompt_template.render(
            user_message=user_message,
            has_session_id=has_session_id,
            has_data_input=has_data_input,
        )
    except Exception:
        return "analyze"  # Safe default if prompt not found

    try:
        content = _invoke_openai_chat(
            settings,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=10,
            timeout=15,
        )
        action = normalize_action(content)
        if action in {"analyze", "revise", "dry-run", "apply", "show"}:
            return action
    except Exception:
        pass  # Fall through to safe default

    return "analyze"  # Safe default


_ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": (
                "Run the full metadata analysis pipeline on uploaded files to generate a fresh proposal. "
                "Use when: new files have been uploaded, there is no existing session, or the user "
                "explicitly wants to start over from scratch."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise",
            "description": (
                "Re-draft the ENTIRE metadata proposal from scratch, incorporating the user's feedback. "
                "Use ONLY when the user wants a complete overhaul — e.g. 'the whole thing is wrong', "
                "'use a completely different style', 'start the description over'. "
                "Never use this for feedback that targets a single field."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_field",
            "description": (
                "Edit exactly ONE metadata field based on the user's feedback. "
                "Use this whenever the user's feedback is narrowly about a single aspect of the metadata: "
                "the title wording or location specificity, the description length or content, "
                "an author name or email, the tags, the license, the date range, etc. "
                "This INCLUDES follow-up refinements when context makes the target field obvious — "
                "e.g. 'zoom in', 'make it shorter', 'more specific', 'actually use X' after a field "
                "was just discussed or updated. Infer the field from context when not stated explicitly. "
                "Prefer this over revise unless the user wants everything redone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "The CKAN metadata field to update. Use the exact CKAN key (e.g. owner_org, "
                            "not 'organization'). Available fields and their aliases are listed in the "
                            "schema field reference appended to this tool's description. "
                            "Infer from context when the user does not name the field explicitly."
                        ),
                    },
                    "instruction": {
                        "type": "string",
                        "description": "What the user wants changed, in their own words.",
                    },
                },
                "required": ["field", "instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dry_run",
            "description": (
                "Validate the current metadata against CKAN without writing anything. "
                "Use when the user asks to validate, preview, check errors, or run a dry run."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply",
            "description": (
                "Register the dataset in CKAN (live write). "
                "Use ONLY when the user explicitly approves registration — e.g. 'REGISTER', 'go ahead', "
                "'create it'. Do not use for questions or uncertainty."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show",
            "description": (
                "Display current metadata without making changes. "
                "Use when the user asks to see the full metadata, OR asks a factual question "
                "about a specific field (e.g. 'what's the title?', 'what org is set?', "
                "'show me the description'). For specific-field questions, set `field` and "
                "`question` so the response is focused rather than a full dump."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "Optional. The specific metadata field the user is asking about. "
                            "Omit to return all fields."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Optional. The user's exact question, to frame the focused answer."
                        ),
                    },
                },
            },
        },
    },
]

_ROUTER_ACTION_MAP = {
    "analyze": "analyze",
    "revise": "revise",
    "revise_field": "revise_field",
    "dry_run": "dry-run",
    "apply": "apply",
    "show": "show",
}


def llm_route_action(
    settings: Settings,
    request: dict[str, Any],
    prior_status: str,
    schema_profile: str = "",
) -> tuple[str, dict[str, Any]]:
    """Use LLM tool-calling to route a natural-language message to an action.

    Returns (action, extra_args). For revise_field, extra_args carries
    {"field": ..., "instruction": ...}. Falls back to "analyze" when the LLM
    is unavailable or returns no tool call.
    """
    if not settings.openai_api_key:
        return "analyze", {}

    message = str(request.get("message") or "").strip()
    if not message:
        return "analyze", {}

    # Build session context from saved state so the router understands what is currently
    # in progress — current title, recently edited fields — without needing conversation history.
    session_context_lines: list[str] = []
    session_id = str(request.get("session_id") or "")
    if session_id and prior_status in {"analyzed", "dry_run", "dry_run_failed", "needs_clarification"}:
        try:
            path = _state_path(settings, session_id)
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                desired = saved.get("desired_dataset_payload") or {}
                origins = saved.get("field_origins") or {}
                if desired.get("title"):
                    session_context_lines.append(f"Current title: {desired['title'][:120]}")
                if desired.get("notes"):
                    session_context_lines.append(f"Current description (excerpt): {str(desired['notes'])[:120]}...")
                user_fields = [k for k, v in origins.items() if v == "user-supplied"]
                if user_fields:
                    session_context_lines.append(f"Fields the user has already edited: {', '.join(user_fields)}")
        except Exception:
            pass

    context_parts = [
        f"User message: {message[:500]}",
        f"Prior workflow status: {prior_status or 'none (first turn)'}",
        f"Has uploaded files: {bool(request.get('files') or request.get('upload_dir') or request.get('upload_dirs') or request.get('source_url') or request.get('source_urls'))}",
    ]
    if session_context_lines:
        context_parts.append("Session state:\n  " + "\n  ".join(session_context_lines))
    context = "\n".join(context_parts)

    system_msg = (
        "You are routing a CKAN metadata registration request to the correct action. "
        "Call the tool that best matches the user's intent given their message and session state. "
        "Rules (in priority order): "
        "(1) When the user challenges or disputes a value — phrasing like 'how could it be X', "
        "'that\\'s wrong', 'that\\'s not right', 'that can\\'t be right', 'that\\'s impossible', "
        "'that date is in the future', 'I\\'m not the maintainer' — call revise_field with "
        "the field containing the disputed value. Never call show for challenge phrasing. "
        "(2) If the user's feedback targets ONE field (including follow-up refinements where the "
        "target field is clear from context), call revise_field — not revise. Only call revise "
        "when the user wants the entire metadata redone. "
        "(3) When the prior status is dry_run and the user expresses approval — 'looks good', "
        "'go ahead', 'proceed', 'submit', 'approved', 'create it', 'sounds good' — call apply."
    )

    # Build schema-aware tool descriptions so the LLM knows field label synonyms.
    router_tools = _ROUTER_TOOLS
    if schema_profile:
        try:
            from app.schemas.registry import SchemaRegistry
            _profile = SchemaRegistry(settings.schemas_dir).get(schema_profile)
            _hints = _profile.field_label_hints()
            if _hints:
                import copy
                router_tools = copy.deepcopy(_ROUTER_TOOLS)
                _field_hint_line = f"Schema field reference: {_hints}"
                for _tool in router_tools:
                    if _tool["function"]["name"] in {"revise_field", "show"}:
                        _tool["function"]["description"] += f" {_field_hint_line}"
        except Exception:
            pass  # Schema unavailable — fall back to generic tool descriptions

    from app import llm as _llm
    try:
        result = _llm.invoke_chat_tools(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": context},
            ],
            router_tools,
            model=settings.ckan_llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "",
            max_tokens=200,
            tool_choice="required",
        )
        calls = result.get("tool_calls") or []
        if calls:
            name = calls[0]["name"]
            args = calls[0].get("arguments") or {}
            action = _ROUTER_ACTION_MAP.get(name)
            if action:
                return action, args
    except Exception as exc:
        logger.warning("[intake] LLM routing failed: %s; defaulting to analyze", exc)

    return "analyze", {}


def make_intake_node(settings: Settings) -> Any:
    def intake(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("intake", state, reason="Parse and normalize incoming request")
        try:
            request = dict(state.get("request") or {})
            thread_id = state_thread_id(request, state.get("thread_id") or uuid.uuid4().hex)
            request["session_id"] = thread_id
            prior_status = str(state.get("status") or "")

            # --- Fast path: explicit action from structured API call ---
            raw_action = state.get("action") or request.get("action") or request.get("command")
            action = normalize_action(raw_action)

            extra_args: dict[str, Any] = {}

            if not action:
                # --- Fast path: explicit REGISTER signal ---
                _msg = str(request.get("message") or "").strip()
                if request.get("approval") or _msg.upper() == APPLY_APPROVAL:
                    action = "apply"
                    if _msg.upper() == APPLY_APPROVAL:
                        request["approval"] = APPLY_APPROVAL

            if not action and prior_status == "needs_dataset_intent":
                # --- Fast path: reply to the "new or update?" dataset-intent prompt ---
                intent = _registration_intent_from_request(request)
                if intent:
                    request["dataset_intent"] = intent
                    action = "dry-run"

            if not action and prior_status in {"dry_run", "dry_run_failed"}:
                # --- Fast path: natural affirmation after dry run → apply ---
                _msg_lower = str(request.get("message") or "").strip().lower()
                if re.search(
                    r"\b(looks?\s+good|go\s+ahead|proceed|create\s+it|approved?|confirm|sounds?\s+good)\b",
                    _msg_lower,
                ):
                    action = "apply"

            if not action:
                # --- LLM routing for natural language messages ---
                action, extra_args = llm_route_action(
                    settings, request, prior_status,
                    schema_profile=str(state.get("schema_profile") or ""),
                )
                action = normalize_action(action)

            if action == "apply" and not request.get("approval"):
                _msg = str(request.get("message") or "").strip()
                if _msg.upper() == APPLY_APPROVAL:
                    request["approval"] = APPLY_APPROVAL

            request["action"] = action
            result: dict[str, Any] = {
                "thread_id": thread_id,
                "request": request,
                "action": action,
                "status": "routed",
            }
            if action == "revise-field" and extra_args:
                result["revise_field_target"] = extra_args
            if action == "show" and extra_args:
                result["show_target"] = extra_args
            elif action == "show":
                # Explicitly clear any prior focused show so stale show_target
                # from a previous turn doesn't bleed into a full-dump request.
                result["show_target"] = {}

            log_node_exit("intake", result, next_node="route")
            return result
        except Exception as e:
            log_error("intake", str(e), state)
            raise

    return intake


def _effective_tapis_token(settings: Settings) -> str:
    """Per-request Tapis JWT (from the chat Authorization header) takes precedence over
    the static env-var token so the logged-in user's identity reaches the MCP server."""
    from app.auth_context import get_request_ckan_auth
    auth = get_request_ckan_auth() or ""
    if auth:
        return auth.removeprefix("Bearer ").removeprefix("bearer ").strip()
    return settings.mcp_tapis_token or ""


def _mcp_get_client(settings: Settings) -> Any:
    """Return the CKAN MCP client if enabled and instantiable, else None.

    The JWT is injected per-call as a ``tapis_token`` tool argument (not in the
    transport headers) so that every call carries the current request's fresh token
    rather than whatever was baked into the shared connection at startup.
    """
    if not settings.mcp_enabled:
        return None
    try:
        from app.tools.mcp_client import get_shared_client

        return get_shared_client(
            settings.mcp_server_url,
            shared_secret=settings.mcp_shared_secret or None,
            timeout=settings.mcp_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP client unavailable: %s", exc)
        return None


# Fields that are internal tracking state only — not valid CKAN package fields.
_CKAN_INTERNAL_FIELDS = frozenset({
    "owner_org_label", "owner_org_name", "owner_org_title", "isopen",
})


def _ckan_metadata_payload(
    desired: dict[str, Any],
    extra_skip: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Strip internal tracking fields and empty values from desired_dataset_payload."""
    skip = _CKAN_INTERNAL_FIELDS | (extra_skip or frozenset())
    return {
        k: v
        for k, v in desired.items()
        if k not in skip and v not in (None, "", [], {})
    }


def _mcp_dry_run(settings: Settings, request: dict[str, Any]) -> dict[str, Any]:
    """Validate package metadata via MCP dry-run and mark saved state as 'dry_run'.

    Calls schema_create_package(dry_run=True) on the MCP server — no CKAN write is made.
    On success, updates the saved state status to 'dry_run' so the apply node accepts it.
    """
    session_id = str(request.get("session_id") or request.get("thread_id") or "")
    path = _state_path(settings, session_id)
    if not path.exists():
        msg = f"No saved state for session `{session_id}`. Analyze files first."
        return {"ok": False, "command": "dry-run", "status": "needs_metadata",
                "message": msg, "review_markdown": msg}

    try:
        saved_state = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "command": "dry-run", "status": "error", "error": str(exc)}

    desired = saved_state.get("desired_dataset_payload") or {}
    resource_plan = saved_state.get("resource_plan") or []
    dataset_type = _clean_metadata_value(desired.get("type")) or "dataset"
    _dry_extra: frozenset[str] = frozenset()
    try:
        from app.schemas.registry import SchemaRegistry
        _dry_sp_name = str(saved_state.get("schema_profile") or "")
        _dry_reg = SchemaRegistry(settings.schemas_dir)
        _dry_sp = _dry_reg.get(_dry_sp_name) if _dry_sp_name else _dry_reg.default()
        _dry_extra = frozenset(_dry_sp.internal_fields)
    except Exception:
        pass
    metadata = _ckan_metadata_payload(desired, _dry_extra)

    client = _mcp_get_client(settings)
    if client is None:
        msg = (
            "MCP server unavailable. Set CKAN_MCP_ENABLED=1 and ensure the MCP server "
            f"is running at {settings.mcp_server_url}."
        )
        return {
            "ok": False,
            "command": "dry-run",
            "status": "error",
            "error": msg,
            "review_markdown": f"## Dry-Run Failed\n\n{msg}",
        }

    _token = _effective_tapis_token(settings) or None
    try:
        pkg_result = client.call_tool("schema_create_package", {
            "dataset_type": dataset_type,
            "metadata": metadata,
            "dry_run": True,
            **({"tapis_token": _token} if _token else {}),
        })
    except Exception as exc:  # noqa: BLE001
        msg = f"MCP call failed: {exc}"
        return {"ok": False, "command": "dry-run", "status": "error", "error": msg,
                "review_markdown": f"## Dry-Run Failed\n\n{msg}"}

    valid = bool(pkg_result.get("valid")) if isinstance(pkg_result, dict) else False
    raw_errors = pkg_result.get("errors") if isinstance(pkg_result, dict) else None
    raw_warnings = pkg_result.get("warnings") if isinstance(pkg_result, dict) else None
    # CKAN returns errors as either a list of strings or a dict {field: [msgs]}.
    errors: list[str] = []
    if isinstance(raw_errors, list):
        errors = [str(e) for e in raw_errors]
    elif isinstance(raw_errors, dict):
        for field_name, msgs in raw_errors.items():
            if isinstance(msgs, list):
                errors += [f"**{field_name}**: {m}" for m in msgs]
            else:
                errors.append(f"**{field_name}**: {msgs}")
    warnings_out: list[str] = []
    if isinstance(raw_warnings, list):
        warnings_out = [str(w) for w in raw_warnings]
    elif isinstance(raw_warnings, dict):
        for field_name, msgs in raw_warnings.items():
            if isinstance(msgs, list):
                warnings_out += [f"**{field_name}**: {m}" for m in msgs]
            else:
                warnings_out.append(f"**{field_name}**: {msgs}")

    lines = ["## CKAN Dry-Run Preview", ""]
    lines += [
        f"**Type**: `{dataset_type}`",
        f"**Name**: `{metadata.get('name')}`",
        f"**Title**: {metadata.get('title')}",
        f"**Organization**: `{metadata.get('owner_org')}`",
        f"**Validation**: {'✓ Valid' if valid else '✗ Invalid'}",
    ]
    if errors:
        lines += ["", "### Errors"]
        lines += [f"- {e}" for e in errors]
    elif not valid and not errors:
        # Surface the raw MCP response so the user can diagnose
        lines += ["", "### Validation details"]
        lines.append(f"```json\n{json.dumps(pkg_result, indent=2)}\n```")
    if warnings_out:
        lines += ["", "### Warnings"]
        lines += [f"- {w}" for w in warnings_out]
    if resource_plan:
        lines += ["", f"### Resources ({len(resource_plan)})"]
        for res in resource_plan:
            if res.get("resource_url"):
                lines.append(f"- `{res.get('resource_name')}` ({res.get('format')}, link → {res.get('resource_url')})")
            else:
                size = _human_bytes(res.get("size_bytes"))
                lines.append(f"- `{res.get('resource_name')}` ({res.get('format')}, {size})")
    lines.append("")
    if valid:
        lines.append("Dry-run passed. Send `REGISTER` to create this dataset and upload resources.")
    else:
        lines.append("Dry-run failed. Fix the errors above before registering.")

    if valid:
        saved_state["status"] = "dry_run"
        saved_state["dry_run_result"] = {"valid": valid, "errors": errors, "warnings": warnings_out}
        _save_existing_state(path, saved_state)

    return {
        "ok": valid,
        "command": "dry-run",
        "status": "dry_run" if valid else "dry_run_failed",
        "valid": valid,
        "errors": errors,
        "warnings": warnings_out,
        "resource_count": len(resource_plan),
        "review_markdown": "\n".join(lines),
        "session_id": session_id,
    }


def _mcp_apply(settings: Settings, request: dict[str, Any]) -> dict[str, Any]:
    """Create the CKAN package and resources via MCP write tools (live write, post-approval).

    The token is sourced from the per-request auth context or settings — never from the model
    or from state — and is NOT written back to any state or log (spec B2 / Fork A).
    """
    session_id = str(request.get("session_id") or request.get("thread_id") or "")
    path = _state_path(settings, session_id)

    try:
        saved_state = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "command": "apply", "status": "error", "error": str(exc)}

    desired = saved_state.get("desired_dataset_payload") or {}
    resource_plan = saved_state.get("resource_plan") or []
    dataset_type = _clean_metadata_value(desired.get("type")) or "dataset"
    _apply_extra: frozenset[str] = frozenset()
    try:
        from app.schemas.registry import SchemaRegistry
        _apply_sp_name = str(saved_state.get("schema_profile") or "")
        _apply_reg = SchemaRegistry(settings.schemas_dir)
        _apply_sp = _apply_reg.get(_apply_sp_name) if _apply_sp_name else _apply_reg.default()
        _apply_extra = frozenset(_apply_sp.internal_fields)
    except Exception:
        pass
    metadata = _ckan_metadata_payload(desired, _apply_extra)

    client = _mcp_get_client(settings)
    if client is None:
        return {"ok": False, "command": "apply", "status": "error",
                "error": "MCP server unavailable.",
                "review_markdown": "## Registration Failed\n\nMCP server is not enabled or reachable."}

    # All CKAN auth goes through Tapis JWTs (no static CKAN API key). Inject the
    # current request's JWT per-call so writes never use a stale cached token.
    _token = _effective_tapis_token(settings) or None
    _tok: dict[str, Any] = {"tapis_token": _token} if _token else {}

    # Create package (live write, approved by human gate before this node runs).
    pkg_args: dict[str, Any] = {
        "dataset_type": dataset_type,
        "metadata": metadata,
        "dry_run": False,
        **_tok,
    }

    def _is_name_conflict(text: str) -> bool:
        t = str(text).lower()
        return "409" in t or "already in use" in t or "that url" in t

    def _try_upsert() -> dict[str, Any] | None:
        """Retry package_create failure as schema_update_package; return result or None on error."""
        logger.info("[apply] package_create name conflict; retrying with schema_update_package")
        upsert_args: dict[str, Any] = {
            "id": metadata.get("name"),
            "metadata_updates": metadata,
            "dry_run": False,
            **_tok,
        }
        try:
            return client.call_tool("schema_update_package", upsert_args)
        except Exception as exc2:  # noqa: BLE001
            logger.warning("[apply] schema_update_package also failed: %s", exc2)
            return None

    # If state indicates an update (user selected an existing dataset), skip create entirely.
    _is_update = (
        saved_state.get("registration_intent") == "update"
        or bool(saved_state.get("existing_ckan_entry"))
    )

    upserted = _is_update
    pkg_result: dict[str, Any] | None = None

    if _is_update:
        update_args: dict[str, Any] = {
            "id": metadata.get("name"),
            "metadata_updates": metadata,
            "dry_run": False,
            **_tok,
        }
        try:
            pkg_result = client.call_tool("schema_update_package", update_args)
        except Exception as exc:  # noqa: BLE001
            msg = f"Package update failed: {exc}"
            return {"ok": False, "command": "apply", "status": "error", "error": msg,
                    "review_markdown": f"## Registration Failed\n\n{msg}"}
    else:
        try:
            pkg_result = client.call_tool("schema_create_package", pkg_args)
        except Exception as exc:  # noqa: BLE001
            if _is_name_conflict(exc):
                pkg_result = _try_upsert()
                upserted = pkg_result is not None
            if not upserted:
                msg = f"Package creation failed: {exc}"
                return {"ok": False, "command": "apply", "status": "error", "error": msg,
                        "review_markdown": f"## Registration Failed\n\n{msg}"}

    # MCP call_tool may return a failure dict instead of raising — handle that too.
    if not isinstance(pkg_result, dict) or not pkg_result.get("success"):
        raw_msg = (pkg_result.get("message") or pkg_result.get("error") or "") \
            if isinstance(pkg_result, dict) else ""
        if not _is_update and _is_name_conflict(raw_msg):
            pkg_result = _try_upsert()
            upserted = isinstance(pkg_result, dict) and bool(pkg_result.get("success"))
        if not isinstance(pkg_result, dict) or not pkg_result.get("success"):
            msg = raw_msg or ("Package update failed" if _is_update else "Package creation failed")
            return {"ok": False, "command": "apply", "status": "error", "error": msg,
                    "review_markdown": f"## Registration Failed\n\n{msg}"}

    package_id: str = str(pkg_result.get("id") or metadata.get("name") or "")
    _upserted = upserted
    dataset_name: str = metadata.get("name", "")
    ckan_url_base = (saved_state.get("ckan") or {}).get("url") or settings.ckan_url
    dataset_url = f"{ckan_url_base.rstrip('/')}/dataset/{dataset_name}"

    # Create resources — one MCP call per entry in resource_plan.
    # Entries with ``local_path`` are uploaded as files (multipart).
    # Entries with ``resource_url`` (no local_path) are registered as CKAN
    # link resources — no bytes transferred, URL stored in CKAN metadata.
    created: list[str] = []
    failed: list[str] = []

    for res in resource_plan:
        local_path = res.get("local_path")
        resource_url = res.get("resource_url")
        if not local_path and not resource_url:
            continue
        resource_meta: dict[str, Any] = {
            "name": res.get("resource_name") or "",
            "description": res.get("resource_description") or "",
            "format": res.get("format") or "",
            "mimetype": res.get("mimetype") or "",
        }
        if resource_url:
            resource_meta["url"] = resource_url
        res_args: dict[str, Any] = {
            "package_id": package_id,
            "resource_metadata": resource_meta,
            "dry_run": False,
            **_tok,
        }
        if local_path:
            res_args["upload_file"] = local_path
        try:
            res_result = client.call_tool("schema_create_resource", res_args)
            if isinstance(res_result, dict) and res_result.get("success"):
                created.append(res.get("resource_name") or local_path or resource_url)
            else:
                err = (res_result.get("message") or res_result.get("error") or "failed") \
                    if isinstance(res_result, dict) else "unknown"
                failed.append(f"{res.get('resource_name')}: {err}")
                logger.warning("Resource %s failed: %s", res.get("resource_name"), err)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{res.get('resource_name')}: {exc}")
            logger.warning("Resource %s exception: %s", res.get("resource_name"), exc)

    # Persist result summary (token is never stored).
    saved_state["status"] = "registered"
    saved_state["registration_result"] = {
        "package_id": package_id,
        "dataset_name": dataset_name,
        "dataset_url": dataset_url,
        "resource_created": len(created),
        "resource_failed": len(failed),
    }
    _save_existing_state(path, saved_state)

    action_label = "updated" if _upserted else "created"
    review_lines = [
        f"## CKAN Registration Complete ({'Updated existing dataset' if _upserted else 'New dataset'})",
        f"- Dataset: `{dataset_name}`",
        f"- URL: {dataset_url}",
        f"- Resources uploaded: `{len(created)}` of `{len(resource_plan)}`",
    ]
    if _upserted:
        review_lines.insert(1, "_Dataset name already existed — metadata was updated in place._")
    if failed:
        review_lines += ["", "### Resource Errors"]
        review_lines += [f"- {e}" for e in failed]

    return {
        "ok": True,
        "command": "apply",
        "status": "registered",
        "dataset_name": dataset_name,
        "dataset_url": dataset_url,
        "package_id": package_id,
        "resource_count": len(resource_plan),
        "resource_created": len(created),
        "resource_updated": 0,
        "resource_removed": 0,
        "resource_errors": failed,
        "review_markdown": "\n".join(review_lines),
        "session_id": session_id,
    }


def make_legacy_command_node(settings: Settings, command: str) -> Any:
    worker = LegacyCkanWorker(settings)

    def run_command(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry(command, state, reason=f"Execute {command} operation")
        request = dict(state.get("request") or {})
        try:
            if command == "dry-run":
                target_issue = _prepare_registration_target(settings, request)
                if target_issue:
                    output = {
                        "result": target_issue,
                        "status": target_issue["status"],
                        "error": target_issue.get("message", ""),
                    }
                    log_node_exit(command, output, next_node="END")
                    return output
            if command in {"dry-run", "apply"}:
                _resolve_owner_org_in_saved_state(settings, request)
            # MCP path: use the running MCP server instead of the legacy worker when enabled.
            if command == "dry-run" and settings.mcp_enabled:
                result = _mcp_dry_run(settings, request)
            else:
                result = worker.run(command, request)
            output = {
                "result": result,
                "status": result.get("status") or result.get("command") or command,
                "error": "",
                "requires_action": None,
            }
            log_node_exit(command, output, next_node="END")
            return output
        except Exception as exc:
            error_output = {
                "result": {"ok": False, "error": str(exc), "command": command},
                "status": "error",
                "error": str(exc),
            }
            log_error(command, str(exc), state)
            return error_output

    return run_command


TEXT_EXTENSIONS = {
    ".c",
    ".cfg",
    ".conf",
    ".csv",
    ".geojson",
    ".html",
    ".htm",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".sql",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TABULAR_EXTENSIONS = {".csv", ".tsv"}
MAX_TEXT_SAMPLE_BYTES = 2_000_000
MAX_DIRECTORY_FILES = 50
MAX_HASH_BYTES = 256 * 1024 * 1024
STATE_SCHEMA_VERSION = 1
NEEDS_INPUT_VALUES = {"needs_user_input", "needs_user_confirmation", "unknown", "none", "n/a"}


def _human_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _slugify(text: str, fallback: str = "uploaded-dataset") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:90].strip("-") or fallback)


def _clean_metadata_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.lower() in NEEDS_INPUT_VALUES:
        return ""
    return text


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _state_path(settings: Settings, session_id: str) -> Path:
    safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(session_id or "").strip()).strip("-")
    return settings.state_dir / f"{safe_session or uuid.uuid4().hex}.json"


def _state_path_from_request(settings: Settings, request: dict[str, Any]) -> Path:
    if request.get("state_path"):
        return Path(str(request["state_path"])).expanduser()
    return _state_path(settings, str(request.get("session_id") or request.get("thread_id") or ""))


def _save_registration_state(settings: Settings, state: dict[str, Any]) -> str:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now_iso()
    path = _state_path(settings, str(state["session_id"]))
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(_json_safe(state), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    return str(path)


def _save_existing_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(_json_safe(state), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _title_from_text(text: str, fallback: str = "Uploaded Dataset") -> str:
    cleaned = re.sub(r"[_\-]+", " ", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:140].title() if cleaned else fallback


def _common_extensions(paths: list[Path]) -> list[str]:
    counts: dict[str, int] = {}
    for path in paths:
        ext = path.suffix.lower() or "[none]"
        counts[ext] = counts.get(ext, 0) + 1
    return [ext for ext, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]]


def _resolve_path(raw_path: str, settings: Settings) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        settings.project_root / path,
        settings.repo_root / path,
        settings.upload_root / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _file_ref_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    for key in ("path", "local_path", "file_path", "filepath", "upload_path", "tmp_path"):
        raw = value.get(key)
        if raw:
            return str(raw)
    nested = value.get("file")
    if isinstance(nested, dict):
        return _file_ref_path(nested)
    return ""


def _file_ref_name(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        return Path(value).name
    if not isinstance(value, dict):
        return fallback
    for key in ("name", "filename", "file_name", "title"):
        raw = value.get(key)
        if raw:
            return str(raw)
    nested = value.get("file")
    if isinstance(nested, dict):
        return _file_ref_name(nested, fallback)
    raw_path = _file_ref_path(value)
    return Path(raw_path).name if raw_path else fallback


def _request_file_references(request: dict[str, Any], settings: Settings) -> tuple[list[dict[str, Any]], list[str]]:
    refs: list[dict[str, Any]] = []
    warnings: list[str] = []

    for field in ("files", "uploaded_files", "attachments"):
        raw_items = request.get(field) or []
        if isinstance(raw_items, (str, dict)):
            raw_items = [raw_items]
        for item in raw_items:
            raw_path = _file_ref_path(item)
            if raw_path:
                refs.append(
                    {
                        "source": field,
                        "path": _resolve_path(raw_path, settings),
                        "display_name": _file_ref_name(item),
                        "description": item.get("description") if isinstance(item, dict) else None,
                    }
                )

    upload_paths: list[Any] = []
    if request.get("upload_dir"):
        upload_paths.append(request["upload_dir"])
    upload_paths.extend(request.get("upload_dirs") or [])
    for raw_path in upload_paths:
        if raw_path:
            refs.append({"source": "upload_dir", "path": _resolve_path(str(raw_path), settings), "display_name": ""})

    expanded: list[dict[str, Any]] = []
    for ref in refs:
        path = ref["path"]
        try:
            if path.is_dir():
                files = sorted(child for child in path.rglob("*") if child.is_file())[: MAX_DIRECTORY_FILES + 1]
                if len(files) > MAX_DIRECTORY_FILES:
                    warnings.append(
                        f"Directory `{path}` has more than {MAX_DIRECTORY_FILES} files; "
                        f"reporting the first {MAX_DIRECTORY_FILES}."
                    )
                    files = files[:MAX_DIRECTORY_FILES]
                expanded.extend(
                    {
                        "source": ref["source"],
                        "path": child,
                        "display_name": str(child.relative_to(path)),
                        "description": ref.get("description"),
                    }
                    for child in files
                )
            else:
                expanded.append(ref)
        except OSError as exc:
            warnings.append(f"Could not inspect `{path}`: {exc}")
            expanded.append(ref)

    return expanded, warnings


def _inline_file_items(request: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for field in ("inline_files", "file_contents"):
        raw_items = request.get(field) or []
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if isinstance(raw_items, str):
            raw_items = [{"name": "pasted_text.txt", "content": raw_items}]
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("text") or item.get("data")
            if content is None:
                continue
            items.append(
                {
                    "name": str(item.get("name") or item.get("filename") or "uploaded_text.txt"),
                    "content": str(content),
                    "mime_type": str(item.get("mime_type") or item.get("content_type") or "text/plain"),
                }
            )
    return items


def _read_text_sample(path: Path, limit: int = MAX_TEXT_SAMPLE_BYTES) -> tuple[str, str, bool]:
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    truncated = len(raw) > limit
    raw = raw[:limit]
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), encoding, truncated
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replacement", truncated


def _count_lines(path: Path) -> int:
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    return line_count


def _sha256(path: Path, size: int) -> str:
    if size > MAX_HASH_BYTES:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preview_lines(text: str, limit: int = 6) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lines.append(line[:180])
        if len(lines) >= limit:
            break
    return lines


def _extract_keywords(text: str) -> list[str]:
    candidates = {
        "ckan",
        "csv",
        "dataset",
        "folium",
        "geojson",
        "geopandas",
        "geospatial",
        "hydrology",
        "jupyter",
        "map",
        "mapping",
        "modflow",
        "netcdf",
        "notebook",
        "pandas",
        "raster",
        "shapefile",
        "tacc",
        "texas",
        "visualization",
    }
    lower = text.lower()
    return sorted(keyword for keyword in candidates if re.search(rf"\b{re.escape(keyword)}\b", lower))


def _text_metadata(path: Path, mime_type: str) -> dict[str, Any]:
    text, encoding, truncated = _read_text_sample(path)
    metadata: dict[str, Any] = {
        "encoding": encoding,
        "sample_truncated": truncated,
        "line_count": _count_lines(path),
        "preview": _preview_lines(text),
        "keywords": _extract_keywords(text),
    }
    if heading_match := re.search(r"(?m)^\s*#{1,6}\s+(.+?)\s*$", text):
        metadata["first_heading"] = heading_match.group(1).strip()[:180]
    urls = sorted(set(re.findall(r"https?://[^\s)\"']+", text)))[:8]
    if urls:
        metadata["url_examples"] = urls
    imports = sorted(
        set(
            match.group(1).split(".")[0]
            for match in re.finditer(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", text)
        )
    )[:20]
    if imports:
        metadata["python_imports"] = imports
    if mime_type in {"text/html", "application/xhtml+xml"} or path.suffix.lower() in {".html", ".htm"}:
        if title_match := re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S):
            metadata["html_title"] = re.sub(r"\s+", " ", title_match.group(1)).strip()[:180]
        if description_match := re.search(
            r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)',
            text,
            re.I,
        ):
            metadata["html_description"] = description_match.group(1).strip()[:300]
    return metadata


def _tabular_metadata(path: Path) -> dict[str, Any]:
    text, _encoding, _truncated = _read_text_sample(path)
    sample = text[:50_000]
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        dialect = csv.excel_tab if delimiter == "\t" else csv.excel

    rows = list(csv.reader(sample.splitlines(), dialect))[:8]
    if not rows:
        return {}
    header = [str(value).strip() for value in rows[0]]
    return {
        "delimiter": delimiter,
        "column_count": len(header),
        "columns": header[:50],
        "estimated_data_rows": max(_count_lines(path) - 1, 0),
        "sample_rows": rows[1:4],
    }


def _json_metadata(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 25 * 1024 * 1024:
        return {"json_note": "File is larger than 25 MB; skipped full JSON parse."}
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if path.suffix.lower() == ".ipynb" and isinstance(data, dict):
        cells = data.get("cells") or []
        code_sources: list[str] = []
        markdown_sources: list[str] = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            source = cell.get("source") or []
            source_text = "".join(source) if isinstance(source, list) else str(source)
            if cell.get("cell_type") == "code":
                code_sources.append(source_text)
            elif cell.get("cell_type") == "markdown":
                markdown_sources.append(source_text)
        code_text = "\n".join(code_sources)
        markdown_text = "\n".join(markdown_sources)
        imports = sorted(
            set(
                match.group(1).split(".")[0]
                for match in re.finditer(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", code_text)
            )
        )[:30]
        headings = [match.group(1).strip()[:160] for match in re.finditer(r"(?m)^\s*#{1,6}\s+(.+)$", markdown_text)]
        return {
            "notebook": {
                "cell_count": len(cells),
                "code_cell_count": len(code_sources),
                "markdown_cell_count": len(markdown_sources),
                "kernelspec": (data.get("metadata") or {}).get("kernelspec") or {},
                "python_imports": imports,
                "markdown_headings": headings[:10],
                "keywords": _extract_keywords(code_text + "\n" + markdown_text),
            }
        }

    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        features = data.get("features") or []
        geometry_types = sorted(
            {
                ((feature.get("geometry") or {}).get("type") or "unknown")
                for feature in features
                if isinstance(feature, dict)
            }
        )
        property_keys: set[str] = set()
        for feature in features[:100]:
            if isinstance(feature, dict) and isinstance(feature.get("properties"), dict):
                property_keys.update(str(key) for key in feature["properties"])
        return {
            "geojson": {
                "feature_count": len(features),
                "geometry_types": geometry_types,
                "property_keys": sorted(property_keys)[:50],
            }
        }

    if isinstance(data, dict):
        return {"json": {"top_level_type": "object", "keys": list(data.keys())[:50]}}
    if isinstance(data, list):
        first_item = data[0] if data else None
        return {
            "json": {
                "top_level_type": "array",
                "item_count": len(data),
                "first_item_keys": list(first_item.keys())[:50] if isinstance(first_item, dict) else [],
            }
        }
    return {"json": {"top_level_type": type(data).__name__}}


def _zip_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        extensions = _common_extensions([Path(name) for name in names])
        return {
            "archive": {
                "file_count": len(names),
                "total_uncompressed_size": sum(info.file_size for info in infos),
                "common_extensions": extensions,
                "sample_files": names[:20],
            }
        }


def _pdf_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}

    # Header metadata — fast, no LLM required.
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        raw_metadata = reader.metadata or {}
        result["pdf"] = {
            "page_count": len(reader.pages),
            "title": str(raw_metadata.get("/Title") or "").strip(),
            "author": str(raw_metadata.get("/Author") or "").strip(),
        }
    except ImportError:
        result["pdf_note"] = "pypdf is not installed; skipped PDF header parse."
    except Exception as exc:
        result["pdf_note"] = f"PDF header parse failed: {exc}"

    # Rich map-reduce summary — reads the full document section-by-section.
    # Silently skipped when the LLM API key is not configured.
    try:
        from app.tools.handlers.pdf import pdf_summarize
        summary = pdf_summarize({"path": str(path)})
        if isinstance(summary, dict) and not summary.get("error"):
            result["pdf_summary"] = summary
    except Exception:
        pass  # LLM unavailable or key not set — header metadata is sufficient fallback

    return result


def _analyze_file(path: Path, display_name: str = "", description: str | None = None) -> dict[str, Any]:
    name = display_name or path.name
    report: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        report["error"] = "File was referenced but is not readable at this path."
        report["metadata_guess"] = {
            "title": _title_from_text(Path(name).stem or name),
            "format": Path(name).suffix.lower().lstrip(".").upper() or "UNKNOWN",
        }
        return report
    if not path.is_file():
        report["error"] = "Referenced path is not a regular file."
        return report

    stat = path.stat()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    extension = path.suffix.lower()
    report.update(
        {
            "size_bytes": stat.st_size,
            "size_human": _human_bytes(stat.st_size),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            "mime_type": mime_type,
            "extension": extension,
            "format": extension.lstrip(".").upper() or mime_type,
            "description": description or "",
        }
    )
    sha256 = _sha256(path, stat.st_size)
    if sha256:
        report["sha256"] = sha256

    try:
        if extension in TABULAR_EXTENSIONS:
            report["tabular"] = _tabular_metadata(path)
        if extension in {".json", ".geojson", ".ipynb"}:
            report.update(_json_metadata(path))
        elif extension == ".zip":
            report.update(_zip_metadata(path))
        elif extension == ".pdf":
            report.update(_pdf_metadata(path))
        if extension in TEXT_EXTENSIONS or mime_type.startswith("text/"):
            report["text"] = _text_metadata(path, mime_type)
    except Exception as exc:
        report["parse_warning"] = str(exc)

    return report


def _analyze_inline_file(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "uploaded_text.txt")
    content = str(item.get("content") or "")
    mime_type = str(item.get("mime_type") or mimetypes.guess_type(name)[0] or "text/plain")
    report: dict[str, Any] = {
        "name": name,
        "exists": True,
        "inline": True,
        "size_bytes": len(content.encode("utf-8")),
        "size_human": _human_bytes(len(content.encode("utf-8"))),
        "mime_type": mime_type,
        "extension": Path(name).suffix.lower(),
        "format": Path(name).suffix.lower().lstrip(".").upper() or mime_type,
        "text": {
            "line_count": len(content.splitlines()),
            "preview": _preview_lines(content),
            "keywords": _extract_keywords(content),
        },
    }
    if report["extension"] in TABULAR_EXTENSIONS:
        try:
            rows = list(csv.reader(content.splitlines()))[:8]
            if rows:
                report["tabular"] = {
                    "column_count": len(rows[0]),
                    "columns": rows[0][:50],
                    "estimated_data_rows": max(len(content.splitlines()) - 1, 0),
                    "sample_rows": rows[1:4],
                }
        except csv.Error as exc:
            report["parse_warning"] = str(exc)
    if report["extension"] in {".json", ".geojson", ".ipynb"}:
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                report["json"] = {"top_level_type": "object", "keys": list(data.keys())[:50]}
            elif isinstance(data, list):
                report["json"] = {"top_level_type": "array", "item_count": len(data)}
        except json.JSONDecodeError as exc:
            report["parse_warning"] = str(exc)
    return report


def _url_reports(request: dict[str, Any]) -> list[dict[str, Any]]:
    raw_urls: list[str] = []
    if request.get("source_url"):
        raw_urls.append(str(request["source_url"]))
    raw_urls.extend(str(url) for url in request.get("source_urls") or [])
    reports: list[dict[str, Any]] = []
    for url in raw_urls:
        parsed = urlparse(url)
        reports.append(
            {
                "url": url,
                "domain": parsed.netloc,
                "path": parsed.path,
                "name": Path(parsed.path).name or parsed.netloc,
                "format": Path(parsed.path).suffix.lower().lstrip(".").upper() or "URL",
            }
        )
    return reports


def _url_report_from_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    return {
        "url": url,
        "domain": parsed.netloc,
        "path": parsed.path,
        "name": Path(parsed.path).name or parsed.netloc,
        "format": Path(parsed.path).suffix.lower().lstrip(".").upper() or "URL",
    }


def _load_saved_metadata_context(
    settings: Settings,
    request: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = _state_path_from_request(settings, request)
    if not path.exists():
        return [], [], [], {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], [], [], {}
    if not isinstance(state, dict):
        return [], [], [], {}

    file_reports = state.get("file_reports") if isinstance(state.get("file_reports"), list) else []
    inline_reports = state.get("inline_file_reports") if isinstance(state.get("inline_file_reports"), list) else []
    source_urls = state.get("source_urls") if isinstance(state.get("source_urls"), list) else []
    url_reports = [_url_report_from_url(str(url)) for url in source_urls if str(url or "").strip()]
    metadata_guess = state.get("metadata_guess") if isinstance(state.get("metadata_guess"), dict) else {}
    return file_reports, inline_reports, url_reports, metadata_guess


def _message_filename_hints(message: str) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    pattern = r"(?<![\w/.-])([A-Za-z0-9_.()\-]+\.(?:csv|json|geojson|ipynb|txt|tsv|zip|pdf|html?))\b"
    for match in re.finditer(pattern, message, re.I):
        filename = match.group(1).strip()
        if filename:
            hints.append(
                {
                    "name": filename,
                    "exists": False,
                    "source": "message",
                    "metadata_guess": {
                        "title": _title_from_text(Path(filename).stem),
                        "format": Path(filename).suffix.lower().lstrip(".").upper(),
                    },
                }
            )
    return hints[:10]


def _resource_description(report: dict[str, Any]) -> str:
    parts: list[str] = []
    if tabular := report.get("tabular"):
        columns = tabular.get("columns") or []
        parts.append(f"Tabular data with {tabular.get('column_count')} column(s)")
        if columns:
            parts.append("columns: " + ", ".join(str(column) for column in columns[:8]))
    if notebook := report.get("notebook"):
        parts.append(
            f"Jupyter notebook with {notebook.get('cell_count')} cell(s), "
            f"including {notebook.get('code_cell_count')} code cell(s)"
        )
    if geojson := report.get("geojson"):
        parts.append(
            f"GeoJSON FeatureCollection with {geojson.get('feature_count')} feature(s)"
        )
    if archive := report.get("archive"):
        parts.append(f"Archive containing {archive.get('file_count')} file(s)")
    if pdf := report.get("pdf"):
        parts.append(f"PDF with {pdf.get('page_count')} page(s)")
    if not parts and (text := report.get("text")):
        preview = text.get("preview") or []
        if preview:
            parts.append(f"Text file beginning with: {preview[0]}")
    return ". ".join(parts)[:500]


def _guess_dataset_metadata(
    request: dict[str, Any],
    file_reports: list[dict[str, Any]],
    inline_reports: list[dict[str, Any]],
    urls: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = request.get("dataset") if isinstance(request.get("dataset"), dict) else {}
    reports = file_reports + inline_reports
    existing_names = [str(report.get("name") or Path(str(report.get("path", ""))).name) for report in reports]
    readable_reports = [report for report in reports if report.get("exists") and not report.get("error")]
    message = str(request.get("message") or "").strip()

    title = str(dataset.get("title") or "").strip()
    if not title:
        for report in readable_reports:
            notebook = report.get("notebook") or {}
            headings = notebook.get("markdown_headings") or []
            text = report.get("text") or {}
            title = str(text.get("first_heading") or (headings[0] if headings else "")).strip()
            if title:
                break
    if not title and existing_names:
        title = _title_from_text(Path(existing_names[0]).stem)
    if not title and urls:
        url_name = Path(str(urls[0].get("name") or "")).stem
        title = _title_from_text(url_name or str(urls[0].get("domain") or "Source Dataset"))
    if not title:
        title = _title_from_text(message[:80], "Uploaded Dataset")

    tags: set[str] = set()
    formats: set[str] = set()
    for report in reports:
        if report.get("format"):
            formats.add(str(report["format"]).lower())
        for section_name in ("text", "notebook"):
            section = report.get(section_name) or {}
            tags.update(str(keyword) for keyword in section.get("keywords") or [])
        if report.get("geojson"):
            tags.update({"geojson", "geospatial"})
        if report.get("tabular"):
            tags.add("tabular")
    for url in urls:
        if url.get("format") and url["format"] != "URL":
            formats.add(str(url["format"]).lower())
        if url.get("domain"):
            tags.add(str(url["domain"]).split(":")[0].replace(".", "-"))
    tags = {_slugify(tag, "") for tag in tags if tag}
    tags.discard("")

    resource_names = existing_names or [str(url.get("name") or url.get("url")) for url in urls]
    resource_phrase = ", ".join(resource_names[:5])
    if dataset.get("notes"):
        notes = str(dataset["notes"])
    elif message and len(message) > 20:
        notes = message[:600]
    elif resource_phrase:
        notes = f"Dataset derived from {len(resource_names)} resource(s): {resource_phrase}."
    else:
        notes = "Starter metadata inferred from the supplied conversation. No readable file bytes were received."

    resources = []
    for report in reports:
        resource_name = str(report.get("name") or Path(str(report.get("path", ""))).name or "resource")
        resources.append(
            {
                "name": _slugify(Path(resource_name).stem, "resource"),
                "title": _title_from_text(Path(resource_name).stem, "Resource"),
                "format": str(report.get("format") or "UNKNOWN"),
                "description": _resource_description(report),
                "path": report.get("path"),
                "size": report.get("size_human"),
            }
        )
    for url in urls:
        resource_name = str(url.get("name") or url.get("domain") or "source")
        resources.append(
            {
                "name": _slugify(Path(resource_name).stem, "source"),
                "title": _title_from_text(Path(resource_name).stem or resource_name, "Source"),
                "format": str(url.get("format") or "URL"),
                "description": f"Source URL from {url.get('domain')}.",
                "url": url.get("url"),
            }
        )

    return {
        "title": title,
        "name": str(dataset.get("name") or _slugify(title)),
        "notes": notes,
        "tags": sorted(tags)[:20],
        "formats": sorted(formats),
        "resources": resources,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _json_for_prompt(value: Any, limit: int = 24_000) -> str:
    text = json.dumps(_json_safe(value), indent=2, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _extract_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Metadata guide response was not a JSON object.")
    return parsed


def _llm_enabled(request: dict[str, Any]) -> bool:
    if request.get("no_llm") is True:
        return False
    return request.get("use_llm") is not False


def _prompt_guided_metadata(
    request: dict[str, Any],
    settings: Settings,
    file_reports: list[dict[str, Any]],
    inline_reports: list[dict[str, Any]],
    url_reports: list[dict[str, Any]],
    metadata_guess: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    prompt_info: dict[str, Any] = {
        "name": "ckan_registration.metadata_guide",
        "path": str(settings.prompt_dir / "ckan_registration" / "metadata_guide.md"),
        "used": False,
    }
    if not _llm_enabled(request):
        prompt_info["reason"] = "LLM disabled for this request."
        return None, prompt_info
    if not settings.openai_api_key:
        prompt_info["reason"] = "OPENAI_API_KEY is not configured; using deterministic metadata guess."
        return None, prompt_info

    try:
        template = PromptRegistry(settings.prompt_dir).load("ckan_registration", "metadata_guide")
    except Exception as exc:
        prompt_info["reason"] = f"metadata_guide prompt could not be loaded: {exc}"
        return None, prompt_info

    evidence = {
        "user_context": request.get("message") or "",
        "conversation_context": request.get("conversation_context") or {},
        "dataset_overrides": request.get("dataset") or {},
        "deterministic_guess": metadata_guess,
        "file_metadata": file_reports + inline_reports,
        "source_urls": url_reports,
        "warnings": warnings,
    }
    _req_profile = None
    try:
        from app.schemas.registry import SchemaRegistry
        _req_sp_name = str(request.get("schema_profile") or "")
        _req_reg = SchemaRegistry(settings.schemas_dir)
        _req_profile = _req_reg.get(_req_sp_name) if _req_sp_name else _req_reg.default()
    except Exception:
        pass
    system_prompt = template.render(
        missing_fields=json.dumps(get_required_metadata_fields(_req_profile)),
        dataset_context=_json_for_prompt(metadata_guess),
        file_metadata=_json_for_prompt(file_reports + inline_reports),
        user_context=str(request.get("message") or ""),
    )
    user_prompt = (
        "Create or revise CKAN starter metadata from the following parsed evidence and conversation context. "
        "If the user is responding to the previous metadata report, preserve prior grounded fields and update "
        "only the fields implicated by the user's reply. "
        "Return strict JSON only, using the structure requested in the metadata guide.\n\n"
        f"{_json_for_prompt(evidence)}"
    )

    try:
        content = _invoke_openai_chat(
            settings,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=3000,
            timeout=45,
        )
        parsed = _extract_json_response(content)
    except Exception as exc:
        prompt_info["reason"] = f"metadata_guide prompt failed: {exc}"
        return None, prompt_info

    prompt_info.update(
        {
            "used": True,
            "model": settings.ckan_llm_model,
            "path": str(template.path),
        }
    )
    return parsed, prompt_info


def _merge_prompt_metadata(
    metadata_guess: dict[str, Any],
    prompt_metadata: dict[str, Any] | None,
    schema_profile: Any = None,
) -> dict[str, Any]:
    if not prompt_metadata:
        return metadata_guess

    merged = dict(metadata_guess)
    package = prompt_metadata.get("ckan_package") or {}
    if isinstance(package, dict):
        if schema_profile is not None and hasattr(schema_profile, "fields"):
            package_fields: tuple[str, ...] = tuple(
                str(f.get("key") or "") for f in schema_profile.fields if f.get("key")
            )
        else:
            package_fields = (
                "title", "name", "notes", "url", "author", "author_email",
                "maintainer", "maintainer_email", "license_id", "version",
                "private", "tags", "spatial", "spatial_description",
                "temporal_coverage_start", "temporal_coverage_end",
            )
        for key in package_fields:
            value = package.get(key)
            if value not in (None, "", [], {}):
                merged[key] = value
        merged["ckan_package"] = package

    resources = prompt_metadata.get("resources")
    if isinstance(resources, list) and resources:
        merged["resources"] = [
            {
                "name": resource.get("name"),
                "title": _title_from_text(str(resource.get("name") or "Resource")),
                "format": resource.get("format") or "UNKNOWN",
                "description": resource.get("description") or resource.get("reason") or "",
                "mimetype": resource.get("mimetype"),
                "resource_type": resource.get("resource_type"),
                "upload_recommendation": resource.get("upload_recommendation"),
            }
            for resource in resources
            if isinstance(resource, dict)
        ]
    return merged


def _tag_dicts(tags: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    raw_tags = tags if isinstance(tags, list) else [tags]
    for tag in raw_tags:
        text = str(tag.get("name") if isinstance(tag, dict) else tag or "")
        name = _slugify(text, "")
        if name and name not in seen:
            seen.add(name)
            out.append({"name": name})
    return out


def _dataset_override(request: dict[str, Any], key: str) -> Any:
    dataset = request.get("dataset") if isinstance(request.get("dataset"), dict) else {}
    return dataset.get(key)


def _ckan_override(request: dict[str, Any], key: str) -> Any:
    ckan = request.get("ckan") if isinstance(request.get("ckan"), dict) else {}
    return ckan.get(key)


def _registration_intent_from_request(request: dict[str, Any]) -> str:
    explicit = _clean_metadata_value(
        request.get("registration_intent")
        or request.get("dataset_intent")
        or request.get("dataset_mode")
        or request.get("ckan_action")
    ).lower()
    if explicit in {"new", "create", "new_dataset", "new-dataset"}:
        return "new"
    if explicit in {"update", "existing", "existing_dataset", "existing-dataset"}:
        return "update"
    if request.get("existing_ckan_entry") or request.get("existing_dataset"):
        return "update"

    message = str(request.get("message") or "").lower()
    if re.search(r"\b(new|create|fresh)\b.*\b(dataset|record|package)\b", message):
        return "new"
    if re.search(r"\b(update|updating|existing|replace|refresh)\b.*\b(dataset|record|package|ckan)\b", message):
        return "update"
    if re.search(r"\b(update|updating|existing|replace|refresh)\b", message):
        return "update"
    return ""


def _existing_dataset_choice_from_request(request: dict[str, Any]) -> str:
    explicit = _clean_metadata_value(
        request.get("existing_ckan_entry")
        or request.get("existing_dataset")
        or request.get("target_dataset")
        or request.get("ckan_dataset")
    )
    if explicit:
        return explicit
    message = str(request.get("message") or "").strip()
    if match := re.search(r"`([^`]+)`", message):
        return match.group(1).strip()
    if match := re.search(r"\b(?:use|choose|select|update)\s+([a-z0-9][a-z0-9_.-]{2,})\b", message, re.I):
        return match.group(1).strip()
    # A bare CKAN name-slug (lowercase, hyphens/underscores, no spaces) is treated as a direct choice.
    slug_candidate = message.strip()
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,}", slug_candidate):
        return slug_candidate
    return ""


def _maintainer_from_auth(ckan_url: str) -> tuple[str, str]:
    """Return (display_name, email) for the logged-in CKAN user, or ('', '') if unavailable."""
    from app.auth_context import get_request_ckan_auth
    auth = get_request_ckan_auth()
    if not auth:
        return "", ""
    try:
        user = CkanClient(base_url=ckan_url, authorization_header=auth).user_show_current()
        if isinstance(user, dict):
            name = _clean_metadata_value(
                user.get("display_name") or user.get("fullname") or user.get("name")
            )
            email = _clean_metadata_value(user.get("email"))
            return name, email
    except Exception:  # noqa: BLE001
        pass
    return "", ""


def _desired_payload_from_guess(
    request: dict[str, Any],
    settings: Settings,
    metadata_guess: dict[str, Any],
) -> dict[str, Any]:
    title = _clean_metadata_value(_dataset_override(request, "title") or metadata_guess.get("title"))
    name = _clean_metadata_value(_dataset_override(request, "name") or metadata_guess.get("name"))
    notes = _clean_metadata_value(_dataset_override(request, "notes") or metadata_guess.get("notes"))
    url = _clean_metadata_value(_dataset_override(request, "url") or metadata_guess.get("url"))
    owner_org = _clean_metadata_value(
        _ckan_override(request, "owner_org")
        or _dataset_override(request, "owner_org")
        or request.get("owner_org")
        or settings.ckan_owner_org
    )

    desired = {
        "name": _slugify(name or title, "ckan-chat-registration-dataset"),
        "title": title or "CKAN Chat Registration Dataset",
        "notes": notes or "Starter CKAN metadata generated from supplied dataset context.",
        "url": url,
        "owner_org": owner_org,
        "private": _parse_bool(_dataset_override(request, "private") or metadata_guess.get("private"), False),
        "tags": _tag_dicts(_dataset_override(request, "tags") or metadata_guess.get("tags") or []),
        "license_id": _clean_metadata_value(
            _dataset_override(request, "license_id") or metadata_guess.get("license_id")
        ),
        "version": _clean_metadata_value(_dataset_override(request, "version") or metadata_guess.get("version")),
        "type": _clean_metadata_value(_dataset_override(request, "type") or metadata_guess.get("type")) or "dataset",
        "isopen": _parse_bool(_dataset_override(request, "isopen") or metadata_guess.get("isopen"), True),
        "spatial": _clean_metadata_value(_dataset_override(request, "spatial") or metadata_guess.get("spatial")),
        "temporal_coverage_start": _clean_metadata_value(
            _dataset_override(request, "temporal_coverage_start") or metadata_guess.get("temporal_coverage_start")
        ),
        "temporal_coverage_end": _clean_metadata_value(
            _dataset_override(request, "temporal_coverage_end") or metadata_guess.get("temporal_coverage_end")
        ),
    }
    desired["author"] = _clean_metadata_value(
        _dataset_override(request, "author") or metadata_guess.get("author")
    )
    desired["author_email"] = _clean_metadata_value(
        _dataset_override(request, "author_email") or metadata_guess.get("author_email")
    )
    # Logged-in CKAN identity is top priority for maintainer; no fallback to caller override.
    ckan_url = _clean_metadata_value(_ckan_override(request, "url") or settings.ckan_url)
    auth_name, auth_email = _maintainer_from_auth(ckan_url)
    desired["maintainer"] = auth_name
    desired["maintainer_email"] = auth_email
    return desired


def _resolve_owner_org(ckan_url: str, owner_org: str, warnings: list[str]) -> dict[str, str]:
    if not owner_org:
        return {"id": "", "name": "", "title": "", "matched_by": "empty"}
    if re.fullmatch(r"[0-9a-fA-F-]{36}", owner_org):
        return {"id": owner_org, "name": owner_org, "title": "", "matched_by": "uuid"}

    try:
        resolved = CkanClient(base_url=ckan_url, timeout=30).resolve_organization_id(owner_org)
    except Exception as exc:
        warnings.append(f"Could not resolve CKAN owner_org `{owner_org}` against `{ckan_url}`: {exc}")
        return {"id": owner_org, "name": owner_org, "title": "", "matched_by": "error"}

    resolved_id = resolved.get("id") or owner_org
    if resolved.get("matched_by") == "unresolved":
        warnings.append(f"Could not find CKAN organization `{owner_org}` in `{ckan_url}`; using it as provided.")
    elif resolved_id != owner_org:
        label = resolved.get("title") or resolved.get("name") or owner_org
        warnings.append(f"Resolved CKAN owner_org `{owner_org}` to `{resolved_id}` ({label}).")
    return resolved


def _resolve_owner_org_in_saved_state(settings: Settings, request: dict[str, Any]) -> None:
    path = _state_path_from_request(settings, request)
    if not path.exists():
        return

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(state, dict):
        return

    desired = state.get("desired_dataset_payload") if isinstance(state.get("desired_dataset_payload"), dict) else {}
    ckan = state.get("ckan") if isinstance(state.get("ckan"), dict) else {}
    ckan_url = _clean_metadata_value(
        _ckan_override(request, "url")
        or request.get("ckan_url")
        or ckan.get("url")
        or settings.ckan_url
    )
    owner_org_label = _clean_metadata_value(
        _ckan_override(request, "owner_org")
        or request.get("owner_org")
        or ckan.get("owner_org_label")
        or desired.get("owner_org_label")
        or desired.get("owner_org")
        or settings.ckan_owner_org
    )
    warnings = state.setdefault("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
        state["warnings"] = warnings

    resolution = _resolve_owner_org(ckan_url, owner_org_label, warnings)
    resolved_id = resolution.get("id") or owner_org_label
    if desired is not None:
        desired["owner_org"] = resolved_id
        desired["owner_org_label"] = owner_org_label
        desired["owner_org_name"] = resolution.get("name") or ""
        desired["owner_org_title"] = resolution.get("title") or ""
    ckan["url"] = ckan_url
    ckan["owner_org"] = resolved_id
    ckan["owner_org_label"] = owner_org_label
    ckan["owner_org_resolution"] = resolution
    state["ckan"] = ckan
    state.setdefault("trace", []).append(
        {
            "step": "ckan.owner_org.resolved",
            "owner_org_label": owner_org_label,
            "owner_org": resolved_id,
            "matched_by": resolution.get("matched_by"),
        }
    )
    _save_existing_state(path, state)


def _dataset_match_score(query: str, dataset: dict[str, Any]) -> float:
    haystack = " ".join(
        str(dataset.get(key) or "")
        for key in ("name", "title", "notes")
    ).lower()
    query_lower = query.lower()
    direct_bonus = 0.25 if query_lower and query_lower in haystack else 0.0
    ratio = difflib.SequenceMatcher(None, query_lower, haystack[: max(len(query_lower) * 3, 120)]).ratio()
    return min(ratio + direct_bonus, 1.0)


def _close_dataset_candidates(ckan_url: str, desired: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    query = _clean_metadata_value(desired.get("title") or desired.get("name"))
    if not query:
        return []
    try:
        results = CkanClient(base_url=ckan_url, timeout=30).package_search(query, rows=10)
    except Exception as exc:
        warnings.append(f"Could not search CKAN for existing dataset matches in `{ckan_url}`: {exc}")
        return []

    candidates = []
    for dataset in results:
        if not isinstance(dataset, dict):
            continue
        score = _dataset_match_score(query, dataset)
        if score < 0.25:
            continue
        org = dataset.get("organization") if isinstance(dataset.get("organization"), dict) else {}
        candidates.append(
            {
                "id": str(dataset.get("id") or ""),
                "name": str(dataset.get("name") or ""),
                "title": str(dataset.get("title") or ""),
                "owner_org": str(dataset.get("owner_org") or org.get("id") or ""),
                "score": round(score, 3),
            }
        )
    return sorted(candidates, key=lambda item: (-float(item["score"]), item["name"]))[:5]


def _dataset_target_markdown(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "Is this a new CKAN dataset, or are you updating an existing one?",
        "",
        "Reply `new dataset` to create a new CKAN package.",
        "Reply `update <dataset-name>` to update an existing package.",
    ]
    if candidates:
        lines.extend(["", "Possible existing matches:"])
        for item in candidates:
            title = item.get("title") or item.get("name")
            lines.append(f"- `{item.get('name')}`: {title}")
    return "\n".join(lines)


def _prepare_registration_target(settings: Settings, request: dict[str, Any]) -> dict[str, Any] | None:
    path = _state_path_from_request(settings, request)
    if not path.exists():
        return None

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(state, dict):
        return None

    desired = state.get("desired_dataset_payload") if isinstance(state.get("desired_dataset_payload"), dict) else {}
    ckan = state.get("ckan") if isinstance(state.get("ckan"), dict) else {}
    warnings = state.setdefault("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
        state["warnings"] = warnings

    intent = _registration_intent_from_request(request) or _clean_metadata_value(state.get("registration_intent"))
    choice = _existing_dataset_choice_from_request(request) or _clean_metadata_value(state.get("existing_ckan_entry"))
    ckan_url = _clean_metadata_value(
        _ckan_override(request, "url")
        or request.get("ckan_url")
        or ckan.get("url")
        or settings.ckan_url
    )

    if choice:
        intent = "update"
        state["existing_ckan_entry"] = choice
        desired["name"] = choice
    if intent == "new":
        state["registration_intent"] = "new"
        state["existing_ckan_entry"] = ""
    elif intent == "update":
        state["registration_intent"] = "update"
        if choice:
            state["existing_ckan_entry"] = choice

    if not intent:
        candidates = _close_dataset_candidates(ckan_url, desired, warnings)
        state["candidate_existing_datasets"] = candidates
        state["status"] = "needs_dataset_intent"
        _save_existing_state(path, state)
        markdown = _dataset_target_markdown(candidates)
        return {
            "ok": False,
            "command": "dry-run",
            "status": "needs_dataset_intent",
            "review_markdown": markdown,
            "message": "Confirm whether this is a new CKAN dataset or an update to an existing one.",
            "candidate_existing_datasets": candidates,
            "session_id": state.get("session_id"),
        }

    if intent == "update" and not state.get("existing_ckan_entry"):
        candidates = _close_dataset_candidates(ckan_url, desired, warnings)
        state["candidate_existing_datasets"] = candidates
        if len(candidates) == 1 and float(candidates[0].get("score") or 0) >= 0.75:
            state["existing_ckan_entry"] = candidates[0]["name"]
            desired["name"] = candidates[0]["name"]
        else:
            state["status"] = "needs_existing_dataset_choice"
            _save_existing_state(path, state)
            markdown = _dataset_target_markdown(candidates)
            return {
                "ok": False,
                "command": "dry-run",
                "status": "needs_existing_dataset_choice",
                "review_markdown": markdown,
                "message": "Choose which existing CKAN dataset should be updated.",
                "candidate_existing_datasets": candidates,
                "session_id": state.get("session_id"),
            }

    state["desired_dataset_payload"] = desired
    state["ckan"] = ckan
    _save_existing_state(path, state)
    return None


def _resource_lookup(resources: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(resources, list):
        return out
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        for key in (resource.get("name"), resource.get("title")):
            name = _slugify(str(key or ""), "")
            if name:
                out[name] = resource
    return out


def _resource_plan_from_reports(
    file_reports: list[dict[str, Any]],
    metadata_guess: dict[str, Any],
) -> list[dict[str, Any]]:
    resources_by_name = _resource_lookup(metadata_guess.get("resources"))
    used_names: set[str] = set()
    plan: list[dict[str, Any]] = []
    source_url = _clean_metadata_value(metadata_guess.get("url"))

    for report in file_reports:
        if not report.get("exists") or report.get("error") or not report.get("path"):
            continue
        path = Path(str(report["path"]))
        if not path.is_file():
            continue

        display_name = str(report.get("name") or path.name)
        base_name = _slugify(Path(display_name).stem or path.stem, "resource")
        resource_name = base_name
        counter = 2
        while resource_name in used_names:
            resource_name = f"{base_name}-{counter}"
            counter += 1
        used_names.add(resource_name)

        prompt_resource = resources_by_name.get(resource_name) or resources_by_name.get(_slugify(display_name, ""))
        description = ""
        if prompt_resource:
            description = _clean_metadata_value(prompt_resource.get("description") or prompt_resource.get("reason"))
        description = description or _resource_description(report) or f"Uploaded file from {display_name}."
        text = report.get("text") if isinstance(report.get("text"), dict) else {}
        keywords = text.get("keywords") or metadata_guess.get("tags") or []

        plan.append(
            {
                "resource_name": resource_name,
                "resource_title": _title_from_text(Path(display_name).stem, "Resource"),
                "resource_description": description,
                "resource_tags": [str(tag) for tag in keywords],
                "source_url": source_url,
                "local_path": str(path),
                "relative_path": display_name,
                "format": str(report.get("format") or path.suffix.lower().lstrip(".").upper() or "BIN"),
                "mimetype": str(
                    report.get("mime_type") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                ),
                "size_bytes": report.get("size_bytes") or path.stat().st_size,
                "sha256": report.get("sha256") or _sha256(path, path.stat().st_size),
                "text_preview": " ".join((text.get("preview") or [])[:3]) if text else "",
            }
        )
    return plan


def _remote_resource_plan_entries(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Build resource plan entries for pre-specified remote URL assets.

    These bypass local file analysis entirely — the caller supplies the URLs and
    optional metadata directly (e.g. WebODM passing orthophoto/DSM/LAZ URLs after
    a processing task). Each entry gets a ``resource_url`` key instead of
    ``local_path`` so the apply node registers them as CKAN link resources
    (no file download or upload).
    """
    raw = request.get("remote_resources") or []
    entries: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for item in raw:
        if isinstance(item, str):
            url, name, fmt, desc = item.strip(), "", "", ""
        elif isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            name = str(item.get("name") or "").strip()
            fmt = str(item.get("format") or "").strip().upper()
            desc = str(item.get("description") or "").strip()
        else:
            continue
        if not url:
            continue

        bare_url = url.split("?")[0]
        if not name:
            name = Path(bare_url).name or url
        if not fmt:
            suffix = Path(bare_url).suffix.lstrip(".").upper()
            fmt = suffix or "URL"

        mimetype = mimetypes.guess_type(bare_url)[0] or "application/octet-stream"
        base_slug = _slugify(Path(name).stem or name, "resource")
        slug = base_slug
        counter = 2
        while slug in used_names:
            slug = f"{base_slug}-{counter}"
            counter += 1
        used_names.add(slug)

        entries.append({
            "resource_name": slug,
            "resource_title": _title_from_text(Path(name).stem or name, "Resource"),
            "resource_description": desc or f"Remote asset: {name}",
            "resource_url": url,
            "format": fmt,
            "mimetype": mimetype,
        })
    return entries


def _metadata_needs_user_input(prompt_metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(prompt_metadata, dict):
        return []
    items = prompt_metadata.get("needs_user_input")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if value not in (None, ""):
        return [str(value)]
    return []


def _save_metadata_registration_state(
    request: dict[str, Any],
    settings: Settings,
    metadata_guess: dict[str, Any],
    prompt_metadata: dict[str, Any] | None,
    file_reports: list[dict[str, Any]],
    inline_reports: list[dict[str, Any]],
    url_reports: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    session_id = str(request.get("session_id") or uuid.uuid4().hex)
    resource_plan = _resource_plan_from_reports(file_reports, metadata_guess)
    resource_plan += _remote_resource_plan_entries(request)
    desired_payload = _desired_payload_from_guess(request, settings, metadata_guess)
    ckan_url = _clean_metadata_value(_ckan_override(request, "url") or settings.ckan_url)
    owner_org_label = _clean_metadata_value(desired_payload.get("owner_org"))
    desired_payload["owner_org_label"] = owner_org_label
    registration_intent = _registration_intent_from_request(request)
    existing_choice = _existing_dataset_choice_from_request(request)
    if existing_choice:
        registration_intent = "update"
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": "metadata_report",
        "registration_intent": registration_intent,
        "message": _clean_metadata_value(request.get("message"))[:3000],
        "source_urls": [url.get("url") for url in url_reports if url.get("url")],
        "existing_ckan_entry": existing_choice,
        "ckan": {
            "url": ckan_url,
            "owner_org": desired_payload.get("owner_org"),
            "owner_org_label": owner_org_label,
            "upload_resources": _parse_bool(request.get("upload_resources"), True),
            "remove_stale_resources": _parse_bool(request.get("remove_stale_resources"), False),
            "resource_extra_fields": _string_list(request.get("resource_extra_fields")),
        },
        "llm_dataset": prompt_metadata or {},
        "metadata_guess": metadata_guess,
        "prompt_metadata": prompt_metadata,
        "needs_user_input": _metadata_needs_user_input(prompt_metadata),
        "desired_dataset_payload": desired_payload,
        "resource_plan": resource_plan,
        "file_reports": file_reports,
        "inline_file_reports": inline_reports,
        "warnings": warnings,
        "trace": [
            {
                "step": "metadata_report.state_saved",
                "resource_count": len(resource_plan),
                "needs_user_input_count": len(_metadata_needs_user_input(prompt_metadata)),
            }
        ],
    }
    state_path = _save_registration_state(settings, state)
    return state_path, desired_payload, resource_plan


def _format_metadata_report(
    metadata_guess: dict[str, Any],
    file_reports: list[dict[str, Any]],
    inline_reports: list[dict[str, Any]],
    url_reports: list[dict[str, Any]],
    warnings: list[str],
    prompt_metadata: dict[str, Any] | None = None,
    prompt_info: dict[str, Any] | None = None,
) -> str:
    lines = ["## File Metadata"]

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)

    all_reports = file_reports + inline_reports
    if not all_reports and not url_reports:
        lines.append("")
        lines.append(
            "I do not see file bytes or a readable local file reference in this request, "
            "so this is a starter guess from the conversation text only."
        )

    for report in all_reports:
        lines.append("")
        name = report.get("name") or Path(str(report.get("path", ""))).name or "unnamed file"
        if report.get("error"):
            lines.append(f"- `{name}`: {report['error']}")
            guess = report.get("metadata_guess") or {}
            if guess:
                lines.append(f"  - Filename-based guess: {guess.get('title')} ({guess.get('format')})")
            continue
        inline_marker = " inline upload" if report.get("inline") else ""
        size = report.get("size_human", "unknown size")
        mime_type = report.get("mime_type", "unknown MIME")
        lines.append(
            f"- `{name}`{inline_marker}: {report.get('format', 'UNKNOWN')} | {size} | {mime_type}"
        )
        if report.get("path"):
            lines.append(f"  - Path: `{report['path']}`")
        if report.get("modified_utc"):
            lines.append(f"  - Modified: `{report['modified_utc']}`")
        if tabular := report.get("tabular"):
            row_count = tabular.get("estimated_data_rows", "unknown")
            column_count = tabular.get("column_count", "unknown")
            lines.append(
                f"  - Table: {row_count} row(s), {column_count} column(s)"
            )
            columns = tabular.get("columns") or []
            if columns:
                lines.append("  - Columns: " + ", ".join(f"`{column}`" for column in columns[:12]))
        if notebook := report.get("notebook"):
            cell_count = notebook.get("cell_count")
            code_count = notebook.get("code_cell_count")
            markdown_count = notebook.get("markdown_cell_count")
            lines.append(
                f"  - Notebook: {cell_count} cell(s), {code_count} code, {markdown_count} markdown"
            )
            imports = notebook.get("python_imports") or []
            if imports:
                lines.append("  - Imports: " + ", ".join(f"`{item}`" for item in imports[:12]))
        if geojson := report.get("geojson"):
            geometry_types = ", ".join(geojson.get("geometry_types") or [])
            lines.append(
                f"  - GeoJSON: {geojson.get('feature_count')} feature(s), geometry: {geometry_types}"
            )
        if archive := report.get("archive"):
            archive_extensions = ", ".join(archive.get("common_extensions") or [])
            lines.append(
                f"  - Archive: {archive.get('file_count')} file(s), common extensions: {archive_extensions}"
            )
        if pdf := report.get("pdf"):
            lines.append(f"  - PDF: {pdf.get('page_count')} page(s)")
        if text := report.get("text"):
            preview = text.get("preview") or []
            if preview:
                lines.append("  - Preview: " + " / ".join(f"`{line}`" for line in preview[:3]))

    for url in url_reports:
        lines.append("")
        lines.append(f"- `{url.get('url')}`: source URL from `{url.get('domain')}`")

    lines.append("")
    lines.append("## Best-Guess CKAN Starter Metadata")
    lines.append(f"- Title: `{metadata_guess.get('title')}`")
    lines.append(f"- Name: `{metadata_guess.get('name')}`")
    notes = str(metadata_guess.get("notes") or "").strip()
    if notes:
        lines.append(f"- Notes: {notes}")
    for field in (
        "url",
        "author",
        "author_email",
        "maintainer",
        "maintainer_email",
        "license_id",
        "version",
        "private",
        "spatial",
        "spatial_description",
        "temporal_coverage_start",
        "temporal_coverage_end",
    ):
        value = metadata_guess.get(field)
        if value not in (None, "", [], {}):
            label = field.replace("_", " ").title()
            lines.append(f"- {label}: `{value}`")
    tags = metadata_guess.get("tags") or []
    if tags:
        lines.append("- Tags: " + ", ".join(f"`{tag}`" for tag in tags))
    formats = metadata_guess.get("formats") or []
    if formats:
        lines.append("- Formats: " + ", ".join(f"`{fmt}`" for fmt in formats))
    resources = metadata_guess.get("resources") or []
    if resources:
        lines.append("- Resources:")
        for resource in resources[:12]:
            detail = resource.get("description") or resource.get("url") or resource.get("path") or ""
            lines.append(f"  - `{resource.get('title')}` ({resource.get('format')}): {detail}")

    if prompt_info:
        lines.append("")
        if prompt_info.get("used"):
            lines.append(f"Metadata guide prompt: `{prompt_info.get('path')}`")
        elif prompt_info.get("reason"):
            lines.append(f"Metadata guide prompt not used: {prompt_info.get('reason')}")

    if prompt_metadata:
        needs_user_input = prompt_metadata.get("needs_user_input") or []
        if needs_user_input:
            lines.append("")
            lines.append("## Needs User Input")
            for item in needs_user_input[:8]:
                if not isinstance(item, dict):
                    continue
                field = item.get("field") or "metadata"
                question = item.get("question") or item.get("why_needed") or "Please provide a value."
                lines.append(f"- `{field}`: {question}")

    lines.append("")
    lines.append("Is this a new CKAN dataset, or are you updating an existing CKAN dataset?")
    lines.append("Reply `new dataset` or `update <dataset-name>` before validation.")
    lines.append("")
    lines.append(
        "Ready to validate this with CKAN? Reply `validate` or `dry run` "
        "to preview CKAN changes before registration."
    )

    return "\n".join(lines)


# Extensions gdalinfo can meaningfully profile via /vsicurl/
_GEO_MCP_EXTENSIONS = frozenset({
    ".tif", ".tiff", ".nc", ".img", ".vrt", ".hdf", ".h5", ".laz", ".las", ".gpkg",
})


def _try_geo_mcp_metadata(
    path: Path,
    settings: Settings,
    upload_id: str,
    display_name: str,
) -> dict[str, Any] | None:
    """Submit a gdalinfo_from_url job for one spatial file; return polled result or None.

    Constructs a temp-file URL served by this API, submits to the Geo MCP actor,
    and polls until complete or poll_timeout is reached.  Never raises.
    """
    if not settings.geo_mcp_enabled or not settings.public_base_url:
        return None
    rel = display_name.lstrip("/") or path.name
    url = f"{settings.public_base_url.rstrip('/')}/v1/uploads/{upload_id}/{rel}"
    try:
        from app.tools.executor import GeoSyncExecutor
        from app.tools.mcp_client import get_shared_client
        client = get_shared_client(
            settings.geo_mcp_url,
            shared_secret=settings.geo_mcp_shared_secret or None,
            timeout=settings.mcp_timeout,
        )
        token = settings.geo_mcp_tapis_token or ""
        sync = GeoSyncExecutor(client, token_value=token, poll_timeout=settings.geo_poll_timeout)
        envelope = sync.invoke("gdalinfo_from_url", {"url": url, "include_stats": False})
        if envelope.get("success"):
            return {"url": url, "result": envelope.get("result")}
        code = (envelope.get("error") or {}).get("code") if isinstance(envelope.get("error"), dict) else None
        if code == "geo_not_ready":
            return {"url": url, "pending": True, "note": "gdalinfo still running; poll execution_id separately"}
    except Exception as exc:
        logger.warning("Geo MCP metadata skipped for %s: %s", path.name, exc)
    return None


def _enrich_spatial_via_geo_mcp(
    file_reports: list[dict[str, Any]],
    file_refs: list[dict[str, Any]],
    settings: Settings,
) -> None:
    """Mutate file_reports in-place: add geo_mcp key for the first spatial file found."""
    if not settings.geo_mcp_enabled or not settings.public_base_url:
        return
    # Extract upload_id from the first file path that lives under upload_root.
    upload_id: str | None = None
    for ref in file_refs:
        try:
            rel = ref["path"].relative_to(settings.upload_root)
            upload_id = rel.parts[0]
            break
        except (ValueError, IndexError, KeyError):
            pass
    if not upload_id:
        return

    submitted = 0
    for report, ref in zip(file_reports, file_refs):
        if submitted >= 1:  # one job per metadata call — avoid long blocking chains
            break
        if not report.get("exists") or report.get("error"):
            continue
        ext = Path(str(ref["path"])).suffix.lower()
        if ext not in _GEO_MCP_EXTENSIONS:
            continue
        display = str(ref.get("display_name") or Path(str(ref["path"])).name)
        geo = _try_geo_mcp_metadata(ref["path"], settings, upload_id, display)
        if geo is not None:
            report["geo_mcp"] = geo
            submitted += 1


def _cleanup_upload_dirs_from_plan(resource_plan: list[dict[str, Any]], settings: Settings) -> None:
    """Remove upload_root subdirs referenced by a resource plan after successful registration."""
    seen: set[str] = set()
    for entry in (resource_plan or []):
        local_path = str(entry.get("local_path") or "")
        if not local_path:
            continue
        try:
            rel = Path(local_path).relative_to(settings.upload_root)
            upload_id = rel.parts[0]
        except (ValueError, IndexError):
            continue
        if upload_id in seen:
            continue
        seen.add(upload_id)
        upload_dir = settings.upload_root / upload_id
        if upload_dir.is_dir():
            shutil.rmtree(upload_dir, ignore_errors=True)
            logger.info("Cleanup A: removed upload dir %s after registration", upload_id)


_FETCHABLE_REMOTE_FORMATS = frozenset({"JSON", "GEOJSON"})
_REMOTE_FETCH_MAX_BYTES = 512 * 1024  # 512 KB — cap so large files are skipped
_REMOTE_FETCH_TIMEOUT = 5  # seconds


def _fetch_remote_json_previews(
    request: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch small JSON/GeoJSON entries from remote_resources and return inline reports.

    Opportunistic enrichment — all errors (network, auth, parse, size) are caught and
    recorded as warnings. Never raises; always returns a (possibly empty) pair.
    """
    import requests as _requests

    raw = request.get("remote_resources") or []
    reports: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        fmt = str(item.get("format") or "").strip().upper()
        name = str(item.get("name") or Path(url.split("?")[0]).name or "remote.json")
        if not url or fmt not in _FETCHABLE_REMOTE_FORMATS:
            continue
        try:
            resp = _requests.get(url, timeout=_REMOTE_FETCH_TIMEOUT, stream=True)
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=8192):
                total += len(chunk)
                if total > _REMOTE_FETCH_MAX_BYTES:
                    warnings.append(
                        f"Remote resource `{name}` exceeds {_REMOTE_FETCH_MAX_BYTES // 1024} KB; skipped."
                    )
                    chunks = []
                    break
                chunks.append(chunk)
            if not chunks:
                continue
            content = b"".join(chunks).decode("utf-8", errors="replace")
            mime = resp.headers.get("content-type", "application/json").split(";")[0].strip()
            report = _analyze_inline_file({"name": name, "content": content, "mime_type": mime})
            report["remote_url"] = url
            reports.append(report)
        except Exception as exc:
            warnings.append(f"Could not fetch remote resource `{name}` for metadata preview: {exc}")

    return reports, warnings


def build_file_metadata_report(request: dict[str, Any], settings: Settings) -> dict[str, Any]:
    file_refs, warnings = _request_file_references(request, settings)
    file_reports = [
        _analyze_file(ref["path"], str(ref.get("display_name") or ""), ref.get("description"))
        for ref in file_refs
    ]
    _enrich_spatial_via_geo_mcp(file_reports, file_refs, settings)
    inline_reports = [_analyze_inline_file(item) for item in _inline_file_items(request)]
    url_reports = _url_reports(request)
    remote_previews, remote_warnings = _fetch_remote_json_previews(request)
    inline_reports.extend(remote_previews)
    warnings.extend(remote_warnings)
    saved_metadata_guess: dict[str, Any] = {}

    if not file_reports and not inline_reports and not url_reports:
        saved_files, saved_inline, saved_urls, saved_metadata_guess = _load_saved_metadata_context(settings, request)
        if saved_files or saved_inline or saved_urls or saved_metadata_guess:
            file_reports = saved_files
            inline_reports = saved_inline
            url_reports = saved_urls

    if not file_reports and not inline_reports and not url_reports:
        file_reports = _message_filename_hints(str(request.get("message") or ""))
        for filename in request.get("attachment_filenames") or []:
            name = str(filename).strip()
            if not name:
                continue
            file_reports.append(
                {
                    "name": name,
                    "exists": False,
                    "source": "attachment",
                    "metadata_guess": {
                        "title": _title_from_text(Path(name).stem),
                        "format": Path(name).suffix.lower().lstrip(".").upper(),
                    },
                }
            )

    metadata_guess = saved_metadata_guess or _guess_dataset_metadata(request, file_reports, inline_reports, url_reports)
    prompt_metadata, prompt_info = _prompt_guided_metadata(
        request,
        settings,
        file_reports,
        inline_reports,
        url_reports,
        metadata_guess,
        warnings,
    )
    _merge_sp = None
    try:
        from app.schemas.registry import SchemaRegistry
        _merge_sp_name = str(request.get("schema_profile") or "")
        _merge_reg = SchemaRegistry(settings.schemas_dir)
        _merge_sp = _merge_reg.get(_merge_sp_name) if _merge_sp_name else _merge_reg.default()
    except Exception:
        pass
    metadata_guess = _merge_prompt_metadata(metadata_guess, prompt_metadata, _merge_sp)
    state_path, desired_payload, resource_plan = _save_metadata_registration_state(
        request,
        settings,
        metadata_guess,
        prompt_metadata,
        file_reports,
        inline_reports,
        url_reports,
        warnings,
    )
    review_markdown = _format_metadata_report(
        metadata_guess,
        file_reports,
        inline_reports,
        url_reports,
        warnings,
        prompt_metadata,
        prompt_info,
    )
    readable_count = sum(
        1 for report in file_reports + inline_reports if report.get("exists") and not report.get("error")
    )

    return {
        "ok": True,
        "command": "metadata-report",
        "status": "metadata_report",
        "message": "Parsed available file metadata and generated a starter CKAN metadata guess.",
        "file_count": len(file_reports) + len(inline_reports),
        "readable_file_count": readable_count,
        "resource_count": len(resource_plan),
        "state_path": state_path,
        "session_id": request.get("session_id"),
        "files": file_reports,
        "inline_files": inline_reports,
        "source_urls": url_reports,
        "warnings": warnings,
        "metadata_guess": metadata_guess,
        "desired_dataset_payload": desired_payload,
        "resource_plan": resource_plan,
        "prompt_metadata": prompt_metadata,
        "metadata_prompt": prompt_info,
        "review_markdown": review_markdown,
    }


def make_file_metadata_node(settings: Settings) -> Any:
    def file_metadata(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("metadata", state, reason="Parse supplied files and report metadata guesses")
        request = dict(state.get("request") or {})
        try:
            result = build_file_metadata_report(request, settings)
            output = {
                "result": result,
                "status": result["status"],
                "action": result["command"],
                "error": "",
                "requires_action": None,
            }
            log_node_exit("metadata", output, next_node="END")
            return output
        except Exception as exc:
            output = {
                "result": {
                    "ok": False,
                    "command": "metadata-report",
                    "status": "error",
                    "error": str(exc),
                },
                "status": "error",
                "error": str(exc),
            }
            log_error("metadata", str(exc), state)
            return output

    return file_metadata


def make_safe_apply_node(settings: Settings) -> Any:
    worker = LegacyCkanWorker(settings)

    def apply_review_markdown(result: dict[str, Any]) -> str:
        dataset_url = result.get("dataset_url") or "<dataset URL unavailable>"
        created = int(result.get("resource_created") or 0)
        updated = int(result.get("resource_updated") or 0)
        removed = int(result.get("resource_removed") or 0)
        total = int(result.get("resource_count") or 0)
        return "\n".join(
            [
                "## CKAN Registration Complete",
                f"- Dataset: `{result.get('dataset_name') or '<unknown>'}`",
                f"- URL: {dataset_url}",
                f"- Resources uploaded/sent to CKAN: `{total}`",
                (
                    f"- Resource create/update/remove counts: `{created}` created, "
                    f"`{updated}` updated, `{removed}` removed"
                ),
            ]
        )

    def apply(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("apply", state, reason="Apply CKAN registration after dry-run and approval")
        request = dict(state.get("request") or {})
        session_id = str(request.get("session_id") or state.get("thread_id") or "")
        path = _state_path(settings, session_id)
        if not path.exists():
            message = f"No saved metadata state found for thread `{session_id}`. Create metadata before registering."
            output = {
                "result": {
                    "ok": False,
                    "command": "apply",
                    "status": "needs_metadata",
                    "message": message,
                    "review_markdown": message,
                },
                "status": "needs_metadata",
                "error": message,
            }
            log_node_exit("apply", output, next_node="END")
            return output

        try:
            saved_state = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            message = f"Could not read saved metadata state for thread `{session_id}`: {exc}"
            output = {
                "result": {"ok": False, "command": "apply", "status": "error", "error": message},
                "status": "error",
                "error": message,
            }
            log_error("apply", message, state)
            return output

        if saved_state.get("status") != "dry_run":
            message = (
                "Run a CKAN dry-run before registering. Ask for `dry run`, review the CKAN changes, "
                "then send `REGISTER` if the dry-run looks correct."
            )
            output = {
                "result": {
                    "ok": False,
                    "command": "apply",
                    "status": "needs_dry_run",
                    "message": message,
                    "review_markdown": message,
                    "session_id": session_id,
                },
                "status": "needs_dry_run",
                "error": message,
            }
            log_node_exit("apply", output, next_node="END")
            return output

        try:
            _resolve_owner_org_in_saved_state(settings, request)
            if settings.mcp_enabled:
                result = _mcp_apply(settings, request)
            else:
                result = worker.run("apply", request)
            if isinstance(result, dict) and not result.get("review_markdown"):
                result["review_markdown"] = apply_review_markdown(result)
            if isinstance(result, dict) and result.get("ok"):
                resource_plan = saved_state.get("resource_plan") or []
                _cleanup_upload_dirs_from_plan(resource_plan, settings)
            output = {
                "result": result,
                "status": result.get("status") or result.get("command") or "apply",
                "error": "",
                "requires_action": None,
            }
            log_node_exit("apply", output, next_node="END")
            return output
        except Exception as exc:
            output = {
                "result": {"ok": False, "error": str(exc), "command": "apply"},
                "status": "error",
                "error": str(exc),
            }
            log_error("apply", str(exc), state)
            return output

    return apply


def make_approval_node() -> Any:
    def approval(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("approval", state, reason="Verify user approval before proceeding to apply")
        request = dict(state.get("request") or {})
        
        if request.get("approval") == APPLY_APPROVAL:
            result = {"request": request, "status": "approved"}
            log_node_exit("approval", result, next_node="apply")
            return result

        log_interrupt(
            "ckan_apply_approval_required",
            "Review the dry-run output and resume with approval exactly equal to REGISTER",
            {"required_approval": APPLY_APPROVAL, "thread_id": state.get("thread_id")}
        )
        resume_payload = interrupt(
            {
                "type": "ckan_apply_approval_required",
                "message": "Review the dry-run output and resume with approval exactly equal to REGISTER.",
                "required_approval": APPLY_APPROVAL,
                "thread_id": state.get("thread_id"),
            }
        )
        if isinstance(resume_payload, dict):
            request.update(resume_payload)
        else:
            request["approval"] = str(resume_payload or "")
        
        result = {
            "request": request,
            "status": "approved" if request.get("approval") == APPLY_APPROVAL else "approval_missing",
        }
        log_node_exit("approval", result, next_node="apply")
        return result

    return approval


# ── Gated geo transform path (spec 2026-06-30) ───────────────────────────────


def _transform_proposal(state: CkanRegistrationState) -> dict[str, Any]:
    """The transform a persona proposed (preferred) or one supplied directly on the request."""
    proposal = state.get("transform_request")
    if isinstance(proposal, dict) and proposal:
        return proposal
    request = dict(state.get("request") or {})
    direct = request.get("transform")
    return direct if isinstance(direct, dict) else {}


def make_geo_approval_node() -> Any:
    """Interrupt for human approval of a proposed geo transform (mirrors the CKAN apply gate).

    The model only *proposes*; this node surfaces the exact effect (operation, destination
    dataset, clip extent) and resumes to geo-apply only on ``approval == REGISTER``.
    """
    from app.agents.ckan_registration.geo_transform import approval_payload

    def geo_approval(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("geo-approval", state, reason="Human approval required before running a geo transform")
        request = dict(state.get("request") or {})
        proposal = _transform_proposal(state)
        # Status polling needs no approval — pass straight through to geo-apply.
        if normalize_action(state.get("action")) == "transform-status":
            result = {"request": request, "transform_request": proposal, "status": "status_check"}
            log_node_exit("geo-approval", result, next_node="geo-apply")
            return result
        if request.get("approval") == APPLY_APPROVAL:
            result = {"request": request, "transform_request": proposal, "status": "approved"}
            log_node_exit("geo-approval", result, next_node="geo-apply")
            return result

        payload = approval_payload(proposal, thread_id=state.get("thread_id"))
        log_interrupt("geo_transform_approval_required", payload["message"], {"thread_id": state.get("thread_id")})
        resume_payload = interrupt(payload)
        if isinstance(resume_payload, dict):
            request.update(resume_payload)
        else:
            request["approval"] = str(resume_payload or "")
        approved = request.get("approval") == APPLY_APPROVAL
        result = {
            "request": request,
            "transform_request": proposal,
            "status": "approved" if approved else "approval_missing",
        }
        log_node_exit("geo-approval", result, next_node="geo-apply")
        return result

    return geo_approval


def _geo_runner(settings: Settings) -> Any:
    """Build a token-injecting GeoTransformRunner, or None when geo/token is not configured."""
    if not settings.geo_mcp_enabled or not settings.geo_mcp_tapis_token:
        return None
    from app.tools import GeoTransformRunner
    from app.tools.mcp_client import get_shared_client

    try:
        client = get_shared_client(
            settings.geo_mcp_url,
            shared_secret=settings.geo_mcp_shared_secret or None,
            timeout=settings.mcp_timeout,
        )
        if not client.ping():
            return None
    except Exception:  # noqa: BLE001
        return None
    return GeoTransformRunner(
        client, token_value=settings.geo_mcp_tapis_token, poll_timeout=settings.geo_poll_timeout
    )


def make_geo_apply_node(settings: Settings) -> Any:
    """Execute an approved geo transform: cap-check, token-inject (server-side), submit, poll.

    The token comes from settings and is injected by the runner — never from the proposal/state,
    and never written back to state (security 2a/2b). Enforces the per-session transform cap.
    """
    from app.agents.ckan_registration.geo_transform import (
        TransformProposalError,
        build_tool_call,
    )

    def _out(status: str, result: dict[str, Any], error: str = "", **extra: Any) -> dict[str, Any]:
        out = {"result": result, "status": status, "error": error}
        out.update(extra)
        return out

    def geo_apply(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("geo-apply", state, reason="Run approved geo transform on the Abaco actor")
        request = dict(state.get("request") or {})

        # Support the transform-status follow-up action without re-running a transform.
        if normalize_action(state.get("action")) == "transform-status":
            runner = _geo_runner(settings)
            execution_id = str(request.get("execution_id") or state.get("transform_execution_id") or "")
            if runner is None:
                return _out("error", {"ok": False, "error": "geo MCP not configured"}, "geo MCP not configured")
            if not execution_id:
                return _out("error", {"ok": False, "error": "no execution_id to poll"}, "no execution_id to poll")
            envelope = runner.poll_status(execution_id)
            ok = bool(envelope.get("success"))
            return _out("ok" if ok else "error", envelope.get("result") or envelope, "" if ok else "poll failed")

        if request.get("approval") != APPLY_APPROVAL:
            msg = "Geo transform not approved. Reply REGISTER to authorize the proposed transform."
            return _out("approval_missing", {"ok": False, "review_markdown": msg, "message": msg}, msg)

        submitted = int(state.get("transforms_submitted") or 0)
        if submitted >= settings.geo_max_transforms_per_session:
            msg = (
                f"Per-session transform limit reached ({settings.geo_max_transforms_per_session}). "
                "Start a new session to run more transforms."
            )
            return _out("limit_reached", {"ok": False, "message": msg, "review_markdown": msg}, msg)

        proposal = _transform_proposal(state)
        try:
            tool_name, args = build_tool_call(proposal)
        except TransformProposalError as exc:
            return _out("error", {"ok": False, "error": str(exc)}, str(exc))

        runner = _geo_runner(settings)
        if runner is None:
            msg = (
                "Geo transforms are unavailable (GEO_MCP_ENABLED/GEO_MCP_TAPIS_TOKEN not "
                "configured or server unreachable)."
            )
            return _out("error", {"ok": False, "error": msg}, msg)

        envelope = runner.run(tool_name, args)
        ok = bool(envelope.get("success"))
        result = envelope.get("result") if ok else {"ok": False, "error": envelope.get("error")}
        if isinstance(result, dict):
            result.setdefault("ok", ok)
            result.setdefault("command", "geo-transform")
        execution_id = ""
        if isinstance(result, dict):
            execution_id = str(result.get("execution_id") or "")
        return _out(
            "ok" if ok else "error",
            result if isinstance(result, dict) else {"ok": ok, "result": result},
            "" if ok else str(envelope.get("error")),
            transforms_submitted=submitted + 1,
            transform_execution_id=execution_id,
        )

    return geo_apply


_REVERSE_GEOCODE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "reverse_geocode",
        "description": (
            "Look up the place name at the dataset's centroid coordinates at a given zoom level. "
            "Use zoom=5 for region/state, zoom=10 for city/town, zoom=14 for suburb/neighbourhood, "
            "zoom=16 for street. Pick the zoom that matches the specificity the user wants for the title."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "zoom": {
                    "type": "integer",
                    "description": (
                        "Nominatim zoom level: 5=region, 8=county, 10=city, "
                        "12=town, 14=suburb, 15=neighbourhood, 16=street, 18=building"
                    ),
                },
            },
            "required": ["zoom"],
        },
    },
}


def _centroid_from_spatial(spatial: str | None) -> tuple[float, float] | None:
    """Extract (lat, lon) centroid from a GeoJSON polygon or bbox string."""
    if not spatial:
        return None
    try:
        geom = json.loads(spatial)
        ring = (geom.get("coordinates") or [[]])[0]
        if ring:
            lats = [p[1] for p in ring if len(p) >= 2]
            lons = [p[0] for p in ring if len(p) >= 2]
            if lats and lons:
                return sum(lats) / len(lats), sum(lons) / len(lons)
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        pass
    import re
    m = re.search(r"W=([-\d.]+).*?E=([-\d.]+).*?S=([-\d.]+).*?N=([-\d.]+)", spatial)
    if m:
        w, e, s, n = map(float, m.groups())
        return (s + n) / 2, (w + e) / 2
    return None


def _nominatim_fetch_raw(lat: float, lon: float, zoom: int, timeout: float = 6.0) -> dict[str, Any] | None:
    """Call Nominatim reverse geocode at a specific zoom; returns raw JSON or None on error."""
    import ssl
    import urllib.request as urlreq

    url = (
        f"https://nominatim.openstreetmap.org/reverse"
        f"?format=json&lat={lat:.6f}&lon={lon:.6f}&zoom={zoom}"
    )
    req = urlreq.Request(url, headers={"User-Agent": "ckan-registration-agent/1.0"})
    for ssl_ctx in (None, "insecure"):
        try:
            kw: dict[str, Any] = {"timeout": timeout}
            if ssl_ctx == "insecure":
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                kw["context"] = ctx
            with urlreq.urlopen(req, **kw) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if ssl_ctx == "insecure":
                return None
    return None


def _update_field_with_llm(
    settings: Settings,
    field: str,
    current_value: Any,
    instruction: str,
    context: dict[str, Any],
) -> Any:
    """Run a targeted LLM call to update a single CKAN metadata field.

    For the title field, uses a two-turn tool-calling exchange: the LLM picks
    a Nominatim zoom level, we call Nominatim with the dataset centroid, then
    the LLM writes the title using the real place name.
    """
    if not settings.openai_api_key:
        return current_value

    context_summary = "\n".join([
        f"Dataset title: {context.get('title', '')}",
        f"Dataset notes: {str(context.get('notes', ''))[:200]}",
    ])

    # --- title: two-turn tool-calling with Nominatim reverse geocode ---
    if field == "title":
        centroid = _centroid_from_spatial(context.get("spatial"))
        if centroid:
            from app import llm as _llm
            lat, lon = centroid
            system_msg = (
                "You are updating a CKAN dataset title. The dataset has known coordinates. "
                "Call reverse_geocode with the zoom level that gives the place-name granularity "
                "matching the user's instruction, then write the new title."
            )
            user_msg = (
                f"Current title: {current_value}\n"
                f"Instruction: {instruction}\n\n"
                "Call reverse_geocode now with the appropriate zoom."
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                turn1 = _llm.invoke_chat_tools(
                    messages,
                    [_REVERSE_GEOCODE_TOOL],
                    model=settings.ckan_llm_model,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url or "",
                    max_tokens=60,
                    tool_choice="required",
                )
                calls = turn1.get("tool_calls") or []
                geo_content = "Geocoding not available — infer from existing title context."
                if calls:
                    zoom = max(1, min(18, int(calls[0].get("arguments", {}).get("zoom", 14))))
                    logger.info("[revise_field] reverse_geocode zoom=%d lat=%.4f lon=%.4f", zoom, lat, lon)
                    geo_data = _nominatim_fetch_raw(lat, lon, zoom)
                    if geo_data:
                        display = geo_data.get("display_name") or ""
                        addr = geo_data.get("address") or {}
                        geo_content = f"display_name: {display}\naddress: {addr}"
                        logger.info("[revise_field] geocoded → %s", display)

                messages_turn2: list[dict[str, Any]] = messages + [turn1["raw_message"]]
                if calls:
                    messages_turn2.append({
                        "role": "tool",
                        "tool_call_id": calls[0]["id"],
                        "content": geo_content,
                    })
                messages_turn2.append({
                    "role": "user",
                    "content": "Write the new dataset title. Reply with ONLY the title string.",
                })
                turn2 = _llm.invoke_chat_tools(
                    messages_turn2,
                    None,
                    model=settings.ckan_llm_model,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url or "",
                    max_tokens=120,
                    tool_choice="auto",
                )
                result = (turn2.get("content") or "").strip()
                if result:
                    return result
            except Exception as exc:
                logger.warning("[revise_field] geocode title path failed: %s — falling back", exc)
        # fall through to plain chat when no centroid or geocode fails

    if field == "tags":
        current_display = ", ".join(
            t.get("name") if isinstance(t, dict) else str(t)
            for t in (current_value or [])
        )
        prompt = (
            f"Update the CKAN tags for this dataset.\n"
            f"Context:\n{context_summary}\n\n"
            f"Current tags: {current_display}\n"
            f"Instruction: {instruction}\n\n"
            f"Reply with ONLY a comma-separated list of tag names. "
            f"Tags must be lowercase with hyphens only (no spaces or special characters)."
        )
    else:
        prompt = (
            f"Update the CKAN metadata field '{field}' for this dataset.\n"
            f"Context:\n{context_summary}\n\n"
            f"Current value: {current_value}\n"
            f"Instruction: {instruction}\n\n"
            f"Reply with ONLY the new value for '{field}', nothing else."
        )

    try:
        response = _invoke_openai_chat(
            settings,
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
            timeout=30,
        )
        text = response.strip()
        if field == "tags":
            parts = [_slugify(t.strip(), "") for t in text.split(",") if t.strip()]
            return [{"name": p} for p in parts if p]
        if field == "name":
            return _slugify(text)
        return text
    except Exception as exc:
        logger.warning("[revise_field] LLM update failed for field %r: %s", field, exc)
        return current_value


def make_revise_field_node(settings: Settings) -> Any:
    def revise_field_node(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("revise-field", state, reason="Update a single metadata field per user instruction")
        request = dict(state.get("request") or {})
        target = state.get("revise_field_target") or {}
        field = str(target.get("field") or "").strip()
        instruction = str(target.get("instruction") or "").strip()
        session_id = str(request.get("session_id") or state.get("thread_id") or "")

        if not field or not instruction:
            msg = "revise_field called without field or instruction"
            log_node_exit("revise-field", {"error": msg}, next_node="END")
            return {"result": {"ok": False, "error": msg, "command": "revise_field"}, "status": "error", "error": msg}

        path = _state_path(settings, session_id)
        if not path.exists():
            msg = f"No saved state for session `{session_id}`. Analyze files first."
            return {"result": {"ok": False, "error": msg, "command": "revise_field", "review_markdown": msg}, "status": "needs_metadata", "error": msg}

        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"result": {"ok": False, "error": str(exc), "command": "revise_field"}, "status": "error", "error": str(exc)}

        desired = dict(saved.get("desired_dataset_payload") or {})
        current_value = desired.get(field)
        new_value = _update_field_with_llm(settings, field, current_value, instruction, desired)
        desired[field] = new_value

        origins = dict(saved.get("field_origins") or {})
        origins[field] = "user-supplied"
        saved["desired_dataset_payload"] = desired
        saved["field_origins"] = origins
        if saved.get("status") not in {"dry_run", "analyzed"}:
            saved["status"] = "analyzed"
        _save_existing_state(path, saved)

        if field == "tags" and isinstance(new_value, list):
            display = ", ".join(t.get("name") if isinstance(t, dict) else str(t) for t in new_value)
        else:
            display = str(new_value or "")

        review_lines = [
            f"## Updated: `{field}`",
            "",
            f"- **{field}** (`user-supplied`): {display}",
            "",
            "Make more changes, ask for a `dry run` to validate, or send `REGISTER` when ready.",
        ]
        result = {
            "ok": True,
            "command": "revise_field",
            "status": "analyzed",
            "session_id": session_id,
            "field_updated": field,
            "review_markdown": "\n".join(review_lines),
        }
        log_node_exit("revise-field", {"field_updated": field}, next_node="END")
        return {"result": result, "status": "analyzed", "error": ""}

    return revise_field_node


def make_show_node(settings: Settings) -> Any:
    """Return current metadata from saved state without touching the legacy worker."""

    def show_node(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("show", state, reason="Return current session metadata")
        request = dict(state.get("request") or {})
        session_id = str(request.get("session_id") or state.get("thread_id") or "")

        path = _state_path(settings, session_id)
        if not path.exists():
            msg = f"No saved state for session `{session_id}`. Analyze files first."
            out = {"result": {"ok": False, "error": msg, "command": "show", "review_markdown": msg},
                   "status": "needs_metadata", "error": msg}
            log_node_exit("show", out, next_node="END")
            return out

        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            msg = f"Could not read saved state: {exc}"
            out = {"result": {"ok": False, "error": msg, "command": "show"}, "status": "error", "error": msg}
            log_node_exit("show", out, next_node="END")
            return out

        desired = dict(saved.get("desired_dataset_payload") or {})
        origins = dict(saved.get("field_origins") or {})
        resource_plan = list(saved.get("resource_plan") or [])
        reviewed_files = list(saved.get("reviewed_files") or [])
        status = saved.get("status") or "analyzed"

        # Focused single-field answer when the router detected a specific question.
        show_target = dict(state.get("show_target") or {})
        if show_target.get("field"):
            field_key = show_target["field"]
            question = show_target.get("question", "")
            # Build alias map from the loaded schema so any schema's labels work
            # without code changes. Falls back to empty dict (exact match still works).
            _field_aliases: dict[str, str] = {}
            try:
                from app.schemas.registry import SchemaRegistry
                _schema_name = str(state.get("schema_profile") or "")
                _reg = SchemaRegistry(settings.schemas_dir)
                _profile = _reg.get(_schema_name) if _schema_name else _reg.default()
                _field_aliases = _profile.label_map()
            except Exception:
                pass
            field_key = _field_aliases.get(field_key.lower(), field_key)
            # Try exact match first, then case-insensitive scan.
            value = desired.get(field_key)
            if value is None:
                for k, v in desired.items():
                    if k.lower() == field_key.lower():
                        field_key = k
                        value = v
                        break
            origin = origins.get(field_key, "llm-derived")
            if value in (None, "", [], {}):
                answer = f"**{field_key}** is not set yet."
            elif field_key == "tags" and isinstance(value, list):
                display = ", ".join(t.get("name") if isinstance(t, dict) else str(t) for t in value)
                answer = f"**{field_key}** (`{origin}`): {display}"
            else:
                answer = f"**{field_key}** (`{origin}`): {value}"
            if question:
                answer = f"{answer}\n\nTo change it, say something like \"update the {field_key}\"."
            focused_result = {
                "ok": True, "command": "show", "status": status,
                "session_id": session_id, "review_markdown": answer,
            }
            out = {"result": focused_result, "status": status, "error": ""}
            log_node_exit("show", out, next_node="END")
            return out

        _show_all_profile = None
        try:
            from app.schemas.registry import SchemaRegistry
            _sa_name = str(state.get("schema_profile") or "")
            _sa_reg = SchemaRegistry(settings.schemas_dir)
            _show_all_profile = _sa_reg.get(_sa_name) if _sa_name else _sa_reg.default()
        except Exception:
            pass
        _SKIP = (
            set(_show_all_profile.internal_fields)
            if _show_all_profile and _show_all_profile.internal_fields
            else {"owner_org_label", "owner_org_name", "owner_org_title", "isopen"}
        )
        lines = ["## Current Metadata"]
        if reviewed_files:
            shown = reviewed_files[:12]
            suffix = f" (+{len(reviewed_files) - 12} more)" if len(reviewed_files) > 12 else ""
            lines.append(f"\nFiles reviewed ({len(reviewed_files)}): {', '.join(shown)}{suffix}")
        lines.append("")
        for k, v in desired.items():
            if k in _SKIP or v in (None, "", [], {}):
                continue
            origin = origins.get(k, "llm-derived")
            if k == "tags" and isinstance(v, list):
                display = ", ".join(t.get("name") if isinstance(t, dict) else str(t) for t in v)
            else:
                display = str(v)
            lines.append(f"- **{k}** (`{origin}`): {display}")

        missing = [k for k, v in desired.items() if k not in _SKIP and v in (None, "", [], {})]
        if missing:
            lines += ["", "**Not set / needs input**"]
            lines += [f"- {k}: not set" for k in missing]

        if resource_plan:
            lines += ["", f"**Resources ({len(resource_plan)})**"]
            for res in resource_plan:
                size = _human_bytes(res.get("size_bytes"))
                lines.append(f"- `{res.get('resource_name')}` ({res.get('format')}, {size})")

        lines += ["", "Make changes, ask for a `dry run` to validate, or send `REGISTER` when ready."]

        result = {
            "ok": True,
            "command": "show",
            "status": status,
            "session_id": session_id,
            "review_markdown": "\n".join(lines),
        }
        out = {"result": result, "status": status, "error": ""}
        log_node_exit("show", out, next_node="END")
        return out

    return show_node


def route_from_intake(state: CkanRegistrationState) -> str:
    action = normalize_action(state.get("action"))
    if action == "revise-field":
        log_routing_decision("intake", "User requested targeted field revision", "revise-field", {"action": action})
        return "revise-field"
    if action == "dry-run":
        log_routing_decision("intake", "User requested CKAN dry-run", "dry-run", {"action": action})
        return "dry-run"
    if action == "apply":
        log_routing_decision("intake", "User requested CKAN registration approval path", "apply", {"action": action})
        return "apply"
    if action in {"geo-transform", "transform-status"}:
        log_routing_decision(
            "intake", f"Action '{action}' uses the gated geo transform path", "geo-transform", {"action": action}
        )
        return "geo-transform"
    if action == "show":
        log_routing_decision("intake", "User requested saved registration state", "show", {"action": action})
        return "show"
    if action in {"analyze", "revise"}:
        log_routing_decision("intake", f"Action '{action}' uses metadata report path", "metadata", {"action": action})
        return "metadata"
    log_routing_decision(
        "intake",
        f"Unknown action '{action}', defaulting to metadata report",
        "metadata",
        {"original_action": action},
    )
    return "metadata"


def has_data_input(request: dict[str, Any]) -> bool:
    """Check if request has data source(s): files, upload paths, URLs, or metadata."""
    return bool(
        request.get("files")
        or request.get("uploaded_files")
        or request.get("upload_dir")
        or request.get("upload_dirs")
        or request.get("source_url")
        or request.get("source_urls")
        or request.get("remote_resources")
        or (request.get("dataset") and request["dataset"].get("title"))
    )


def has_session_id(request: dict[str, Any]) -> bool:
    """Check if request has a valid session_id for resuming work."""
    return bool(request.get("session_id") and str(request["session_id"]).strip())


def get_required_metadata_fields(schema_profile: Any = None) -> list[str]:
    """Get list of required metadata field names, derived from schema profile when available."""
    if schema_profile is not None and hasattr(schema_profile, "fields"):
        return [str(f.get("key") or "") for f in schema_profile.fields if f.get("required") and f.get("key")]
    return [
        "title", "name", "notes", "author", "author_email",
        "maintainer", "maintainer_email", "license_id"
    ]


def get_metadata_field_guidance() -> dict[str, dict[str, str]]:
    """Guidance for each required metadata field."""
    return {
        "title": {
            "description": "Human-readable name for your dataset",
            "rules": "Descriptive but concise (under 140 characters)",
            "example": "Folium Interactive Maps of Texas",
        },
        "name": {
            "description": "CKAN-friendly identifier (machine-readable slug)",
            "rules": "Lowercase only, hyphens replace spaces, no special characters",
            "example": "folium-texas-maps",
        },
        "notes": {
            "description": "Dataset description",
            "rules": "1-2 sentences explaining content, source, or purpose",
            "example": (
                "Interactive maps created with Folium showing geographic features. "
                "Generated from satellite imagery."
            ),
        },
        "author": {
            "description": "Person or organization that created the dataset",
            "rules": "Name only, no email",
            "example": "Dr. Jane Smith",
        },
        "author_email": {
            "description": "Email contact for the dataset author",
            "rules": "Valid email address",
            "example": "jsmith@example.edu",
        },
        "maintainer": {
            "description": "Person or organization maintaining the dataset",
            "rules": "Can be same as author or a designated curator",
            "example": "Data Services Team",
        },
        "maintainer_email": {
            "description": "Email contact for the maintainer",
            "rules": "Valid email address",
            "example": "dataservices@example.edu",
        },
        "license_id": {
            "description": "CKAN license identifier (must be valid)",
            "rules": "Valid CKAN license: cc-by, cc-by-sa, cc0, odc-by, odc-odbl, cc-nc",
            "example": "cc-by",
        },
    }


def format_metadata_guidance(missing_fields: list[str]) -> str:
    """Format guidance for missing metadata fields."""
    guidance = get_metadata_field_guidance()
    lines = [
        "To complete your dataset registration, I need the following information:",
        "",
    ]
    
    for field in missing_fields:
        if field in guidance:
            info = guidance[field]
            lines.append(f"**{field}**: {info['description']}")
            lines.append(f"   - {info['rules']}")
            lines.append(f"   - Example: {info['example']}")
        else:
            lines.append(f"**{field}**: [field guidance not available]")
        lines.append("")
    
    return "\n".join(lines)


def validate_metadata(
    dataset: dict[str, Any] | None,
    schema_profile: Any = None,
) -> tuple[bool, list[str]]:
    """Validate that dataset metadata has all required fields.

    Returns: (is_valid, missing_fields)
    """
    if not dataset or not isinstance(dataset, dict):
        return False, get_required_metadata_fields(schema_profile)

    missing = []
    for field in get_required_metadata_fields(schema_profile):
        value = dataset.get(field)
        if not value or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    
    return len(missing) == 0, missing


def has_minimal_metadata(dataset: dict[str, Any] | None) -> bool:
    """Check if dataset has MINIMAL metadata to start analysis.
    
    Minimal = at least title OR name (can infer others later).
    """
    if not dataset or not isinstance(dataset, dict):
        return False
    
    title = dataset.get("title") or ""
    name = dataset.get("name") or ""
    return bool((title and str(title).strip()) or (name and str(name).strip()))


def make_plan_node() -> Any:
    """Plan and clarify inputs before routing to action nodes.
    
    Uses LLM to classify user intent if action is not explicitly provided.
    Validates that required inputs are present for the inferred action.
    Interrupts if inputs are missing, asking the user for clarification.
    """
    def plan(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("plan", state, reason="Validate inputs and determine routing strategy")
        request = dict(state.get("request") or {})
        settings = Settings.from_env()
        
        # Check what we have
        has_data = has_data_input(request)
        has_session = has_session_id(request)
        message = str(request.get("message") or "").strip()
        dataset = request.get("dataset") or {}
        metadata_valid, missing_fields = validate_metadata(dataset)
        
        logger.debug(
            f"   Inputs available: has_data={has_data}, has_session={has_session}, "
            f"has_message={bool(message)}, metadata_valid={metadata_valid}"
        )
        
        # Infer action: explicit first, then LLM if needed
        action = normalize_action(state.get("action"))
        if action not in {"analyze", "revise", "dry-run", "apply", "show"}:
            # Use LLM to classify the user's intent
            logger.debug(f"   Action not explicit (got '{action}'), using LLM to classify intent")
            action = llm_classify_action(settings, message, has_session, has_data)
            logger.debug(f"   LLM classified action as: {action}")
        else:
            logger.debug(f"   Explicit action provided: {action}")
        
        request["action"] = action

        # SHOW or REVISE require an existing session
        if action in {"show", "revise"}:
            if not has_session:
                logger.debug(f"   Action '{action}' requires session_id, but none provided")
                log_interrupt(
                    "clarification_required",
                    f"Cannot {action} without a session_id. Please provide a session_id or start a new analysis.",
                    {"action": action, "thread_id": state.get("thread_id")}
                )
                resume_payload = interrupt(
                    {
                        "type": "clarification_required",
                        "action": action,
                        "message": (
                            f"Cannot {action} without a session_id. "
                            "Please provide a session_id or start a new analysis."
                        ),
                        "thread_id": state.get("thread_id"),
                    }
                )
                if isinstance(resume_payload, dict):
                    request.update(resume_payload)
                result = {"request": request, "status": "awaiting_clarification", "action": "analyze"}
                log_node_exit("plan", result, next_node="plan")
                return result
            logger.debug(f"   Session found, proceeding with action '{action}'")
            result = {"request": request, "status": "ready", "clarified": True, "action": action}
            log_node_exit("plan", result, next_node=action)
            return result

        # ANALYZE or DRY-RUN require data input + some metadata context
        if action in {"analyze", "dry-run"}:
            # Check for data input first - this is REQUIRED
            if not has_data:
                logger.debug(f"   Action '{action}' requires data input, but none provided")
                log_interrupt(
                    "clarification_required",
                    f"To {action}, I need to know where your data is.",
                    {"action": action, "required_fields": ["upload_dir", "upload_dirs", "source_url", "source_urls"]}
                )
                resume_payload = interrupt(
                    {
                        "type": "clarification_required",
                        "action": action,
                        "message": (
                            f"To {action}, I need to know where your data is. Please provide at least ONE of: "
                            "readable local file path (`upload_dir`), upload directory (`upload_dirs`), "
                            "source URL (`source_url`, `source_urls`)."
                        ),
                        "required_fields": ["upload_dir", "upload_dirs", "source_url", "source_urls"],
                        "thread_id": state.get("thread_id"),
                    }
                )
                if isinstance(resume_payload, dict):
                    request.update(resume_payload)
                result = {"request": request, "status": "awaiting_clarification", "action": action}
                log_node_exit("plan", result, next_node="plan")
                return result
            
            logger.debug("   Data source found, checking metadata context")
            
            # Check for MINIMAL metadata context (title/name) OR rich message context
            has_minimal_metadata_info = has_minimal_metadata(dataset)
            has_rich_message = message and len(message) > 20  # Enough context in message
            
            if not has_minimal_metadata_info and not has_rich_message:
                # User has data but no metadata info - ask for at least basic context
                logger.debug("   Data present but lacking metadata context")
                log_interrupt(
                    "clarification_required",
                    "Missing dataset metadata: need title/name and description",
                    {
                        "action": action,
                        "has_minimal_metadata": has_minimal_metadata_info,
                        "has_rich_message": has_rich_message,
                    },
                )
                resume_payload = interrupt(
                    {
                        "type": "clarification_required",
                        "action": action,
                        "message": (
                            "I found your data, but I need basic information about it. Please provide at least: "
                            "a dataset title or name, and a brief description (1-2 sentences) of what this "
                            "dataset contains."
                        ),
                        "required_fields": ["title_or_name", "notes_or_description"],
                        "example": {
                            "dataset": {
                                "title": "Folium Mapping Dataset",
                                "name": "folium-mapping",
                                "notes": "Interactive maps created with Folium showing geographic features."
                            },
                            "message": (
                                "This is a collection of interactive maps I created to visualize geospatial data."
                            ),
                        },
                        "thread_id": state.get("thread_id"),
                    }
                )
                if isinstance(resume_payload, dict):
                    request.update(resume_payload)
                result = {"request": request, "status": "awaiting_clarification", "action": action}
                log_node_exit("plan", result, next_node="plan")
                return result
            
            logger.debug(
                "   Data + context complete, ready to analyze "
                f"(has_metadata={has_minimal_metadata_info}, has_rich_msg={has_rich_message})"
            )
            # User has data + some context (minimal metadata OR rich message)
            # Proceed to analyze - LLM will handle iterating on metadata
            result = {"request": request, "status": "ready", "clarified": True, "action": action}
            log_node_exit("plan", result, next_node=action)
            return result

        # APPLY routes to approval, which will handle approval logic
        if action == "apply":
            logger.debug("   Action is 'apply', routing to approval node")
            result = {"request": request, "status": "ready", "clarified": True, "action": action}
            log_node_exit("plan", result, next_node="approval")
            return result

        # Fallback for unknown actions
        logger.debug(f"   Fallback: unknown action '{action}', defaulting to ready state")
        result = {"request": request, "status": "ready", "clarified": True, "action": action}
        log_node_exit("plan", result, next_node="unknown")
        return result

    return plan


def route_from_plan(state: CkanRegistrationState) -> str:
    """Route from planning node to action or back to awaiting input."""
    if state.get("status") == "awaiting_clarification":
        reason = "Waiting for user to provide clarification on missing inputs"
        log_routing_decision("plan", reason, "plan", {"awaiting_fields": state.get("awaiting_fields", [])})
        return "plan"  # Loop back to plan if still awaiting input
    
    action = normalize_action(state.get("action"))
    if action in {"analyze", "revise", "dry-run", "apply", "show"}:
        reason = f"User has provided necessary inputs for action '{action}'"
        log_routing_decision("plan", reason, action, {"action": action, "status": state.get("status")})
        return action
    
    log_routing_decision(
        "plan",
        f"Unknown action '{action}', defaulting to analyze",
        "analyze",
        {"original_action": action},
    )
    return "analyze"
