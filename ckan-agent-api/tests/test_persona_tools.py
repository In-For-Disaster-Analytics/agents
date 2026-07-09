"""Tool-calling engine loop + persona tools wiring (spec Increment 6c)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from app.agents.ckan_registration.persona_nodes import make_persona_node
from app.personas import PersonaRegistry, run_persona_metadata_loop
from app.schemas import SchemaRegistry
from app.settings import PROJECT_ROOT, get_settings

SEED_PERSONAS_DIR = PROJECT_ROOT / "app" / "personas"
SEED_SCHEMAS_DIR = PROJECT_ROOT / "app" / "schemas"

AUTHOR_TOOLS = [{"type": "function", "function": {"name": "file_profile_csv", "parameters": {}}}]
FINAL_JSON = json.dumps({"title": "T", "name": "t", "notes": "n"})


def _personas():
    reg = PersonaRegistry(SEED_PERSONAS_DIR)
    return reg.author(), reg.evaluators()


def _subside():
    return SchemaRegistry(SEED_SCHEMAS_DIR).get("subside")


def _pass_evaluators(system, payload):
    return json.dumps({"verdict": "pass", "questions": []})


class _CountingExec:
    def __init__(self):
        self.calls = []

    def invoke(self, name, args):
        self.calls.append((name, args))
        return {"success": True, "tool": name, "result": {"ok": True}}


def test_author_tool_loop_executes_tool_then_finalizes():
    author, evaluators = _personas()
    executor = _CountingExec()
    turns = []

    def tool_chat(messages, tools):
        turns.append(tools is not None)
        if len(turns) == 1:
            return {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "file_profile_csv", "arguments": {"path": "/x.csv"}}],
                "raw_message": {"role": "assistant", "content": None},
            }
        return {"content": FINAL_JSON, "tool_calls": [], "raw_message": {}}

    result = run_persona_metadata_loop(
        {"source": "x"},
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_subside(),
        chat_fn=_pass_evaluators,
        tool_executor=executor,
        author_tool_specs=AUTHOR_TOOLS,
        tool_chat_fn=tool_chat,
    )
    assert result.converged is True
    assert result.proposed_metadata["name"] == "t"
    assert executor.calls == [("file_profile_csv", {"path": "/x.csv"})]


def test_tool_loop_respects_max_tool_calls_cap():
    author, evaluators = _personas()
    executor = _CountingExec()
    call_counter = {"n": 0}

    def tool_chat(messages, tools):
        if tools is None:  # forced final turn after budget exhausted
            return {"content": FINAL_JSON, "tool_calls": []}
        # Use a distinct path each turn so dedup cache never fires — this isolates
        # the budget-cap behaviour from the duplicate-suppression behaviour.
        call_counter["n"] += 1
        return {
            "content": None,
            "tool_calls": [{"id": "c", "name": "file_profile_csv",
                            "arguments": {"path": f"/x{call_counter['n']}.csv"}}],
            "raw_message": {"role": "assistant", "content": None},
        }

    result = run_persona_metadata_loop(
        {"source": "x"},
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_subside(),
        chat_fn=_pass_evaluators,
        tool_executor=executor,
        author_tool_specs=AUTHOR_TOOLS,
        tool_chat_fn=tool_chat,
        max_tool_calls=2,
    )
    assert result.converged is True
    assert len(executor.calls) == 2  # capped at max_tool_calls


def test_no_tool_params_keeps_simple_author_path():
    author, evaluators = _personas()

    def chat(system, payload):
        if "Domain Expert" in system:
            return FINAL_JSON
        return json.dumps({"verdict": "pass", "questions": []})

    result = run_persona_metadata_loop(
        {"source": "x"},
        author_persona=author,
        evaluator_personas=evaluators,
        schema_profile=_subside(),
        chat_fn=chat,
    )
    assert result.converged is True


def test_seed_author_declares_tools():
    author, _ = _personas()
    assert "ckan_package_search" in author.tools
    assert "file_profile_csv" in author.tools


@dataclasses.dataclass
class _FakeResult:
    proposed_metadata: dict
    clarification_questions: list
    stop_reason: str
    transcript: list = dataclasses.field(default_factory=list)


def _run_node(persona_tools_enabled: bool):
    captured = {}

    def fake_engine(consolidated, **kw):
        captured.update(kw)
        return _FakeResult({"title": "T", "name": "t"}, [], "converged")

    settings = dataclasses.replace(get_settings(), persona_tools_enabled=persona_tools_enabled)
    node = make_persona_node(settings, engine=fake_engine)
    node({"thread_id": "t", "request": {"session_id": "t", "message": "hi"}})
    return captured


def test_node_wires_tools_when_enabled():
    captured = _run_node(True)
    assert captured.get("tool_executor") is not None
    assert captured.get("author_tool_specs")  # non-empty allow-list resolved from the catalog


def test_node_omits_tools_when_disabled():
    captured = _run_node(False)
    assert "tool_executor" not in captured
