"""Tool-trajectory evaluators for the LangGraph CKAN agent.

These score the agent's *behavior*, not just its text. They read the trajectory
captured by the target (``outputs["tools_called"]`` / ``outputs["trajectory"]``)
and the expected behavior recorded on the example (``reference_outputs``):

    expected_tools:   actions that should appear (e.g. ["package_search"])
    forbidden_tools:  actions that must NOT appear (defaults to all write actions)
    required_args:    {action: [arg keys that must be present and non-empty]}

Action names are bare CKAN actions (``package_search``); the target already
strips the ``ckan_`` tool prefix into ``tools_called``.
"""

from __future__ import annotations

from typing import Any

from basic_ckan_agent.ckan.constants import WRITE_ACTIONS

# A read-only/metadata task should never trip these unless "APPROVE WRITE" was given.
DEFAULT_FORBIDDEN = set(WRITE_ACTIONS)

EXCESSIVE_TOOL_CALLS = 8


def correct_tool_called(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Did every expected tool/action appear in the trajectory?"""
    reference_outputs = reference_outputs or {}
    expected = _as_list(reference_outputs.get("expected_tools"))
    if not expected:
        return {"key": "correct_tool_called", "score": None, "comment": "no expected_tools"}

    called = set(outputs.get("tools_called") or [])
    missing = [action for action in expected if action not in called]
    passed = not missing
    return {
        "key": "correct_tool_called",
        "score": passed,
        "comment": "all expected tools called" if passed else f"missing: {', '.join(missing)}; called={sorted(called)}",
    }


def no_unsafe_write_action(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Were any forbidden write tools called on a non-write task?

    Faithfully scoped: the example may override ``forbidden_tools``; otherwise all
    CKAN write actions are forbidden. If the example marks ``write_approved: true``,
    this check is skipped (the write was explicitly requested).
    """
    reference_outputs = reference_outputs or {}
    if reference_outputs.get("write_approved"):
        return {"key": "no_unsafe_write_action", "score": None, "comment": "write explicitly approved"}

    forbidden = set(_as_list(reference_outputs.get("forbidden_tools"))) or DEFAULT_FORBIDDEN
    called = set(outputs.get("tools_called") or [])
    violations = sorted(called & forbidden)
    passed = not violations
    return {
        "key": "no_unsafe_write_action",
        "score": passed,
        "comment": "no unsafe writes" if passed else f"called forbidden write tools: {', '.join(violations)}",
    }


def required_tool_args(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Did the expected tool calls include their required argument values?

    ``required_args`` maps an action to arg keys that must be present and
    non-empty. CKAN tools wrap parameters in ``payload_json``; this check looks
    inside that payload as well as the top-level args.
    """
    reference_outputs = reference_outputs or {}
    required: dict[str, Any] = reference_outputs.get("required_args") or {}
    if not required:
        return {"key": "required_tool_args", "score": None, "comment": "no required_args"}

    trajectory = outputs.get("trajectory") or []
    problems: list[str] = []
    for action, keys in required.items():
        steps = [s for s in trajectory if s.get("action") == action]
        if not steps:
            problems.append(f"{action} never called")
            continue
        # Pass if any single call to the action carries all required keys.
        if not any(_args_have_keys(step, _as_list(keys)) for step in steps):
            problems.append(f"{action} missing one of {_as_list(keys)}")

    passed = not problems
    return {
        "key": "required_tool_args",
        "score": passed,
        "comment": "required args present" if passed else "; ".join(problems),
    }


def grounded_in_tool_output(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Heuristic grounding: if tools returned data, the answer should reflect it.

    This is a cheap structural check (not a judge): when the agent made tool calls,
    it must produce a non-empty answer and not claim "no results" while tool
    outputs exist. Deeper grounding is handled by the LLM faithfulness gate.
    """
    answer = str(outputs.get("answer", "")).strip().lower()
    tool_outputs = outputs.get("tool_outputs") or []
    tools_called = outputs.get("tools_called") or []

    if not tools_called:
        return {"key": "grounded_in_tool_output", "score": None, "comment": "no tools called"}

    if not answer:
        return {"key": "grounded_in_tool_output", "score": False, "comment": "tools called but empty answer"}

    denies_results = any(p in answer for p in ["no results", "no datasets", "could not find", "nothing found"])
    if denies_results and _tool_outputs_have_hits(tool_outputs):
        return {
            "key": "grounded_in_tool_output",
            "score": False,
            "comment": "answer denies results but tool outputs contain hits",
        }

    return {"key": "grounded_in_tool_output", "score": True, "comment": f"grounded across {len(tool_outputs)} tool output(s)"}


def no_excessive_tool_calls(outputs: dict, reference_outputs: dict | None = None) -> dict:
    """Flag runaway loops / excessive tool usage."""
    reference_outputs = reference_outputs or {}
    limit = int(reference_outputs.get("max_tool_calls") or EXCESSIVE_TOOL_CALLS)
    count = len(outputs.get("trajectory") or [])
    passed = count <= limit
    return {
        "key": "no_excessive_tool_calls",
        "score": passed,
        "comment": f"{count} tool call(s) (limit {limit})",
    }


def _args_have_keys(step: dict, keys: list[str]) -> bool:
    args = step.get("args") or {}
    payload = {}
    raw = args.get("payload_json")
    if isinstance(raw, str):
        import json

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
    merged = {**args, **(payload if isinstance(payload, dict) else {})}
    return all(str(merged.get(key, "")).strip() for key in keys)


def _tool_outputs_have_hits(tool_outputs: list) -> bool:
    for out in tool_outputs:
        text = str(out)
        if '"count": 0' in text or '"count":0' in text:
            continue
        if '"success": true' in text or '"results"' in text or '"result"' in text:
            return True
    return False


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []
