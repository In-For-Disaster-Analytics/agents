"""Helpers to extract structured signals from an agent turn.

The agent answers in prose, optionally ending with a ```json block carrying the
proposed metadata. These helpers pull out the generated title/description and the
tool-call trajectory so evaluators can score them without parsing free text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AnyMessage

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_metadata(answer: str) -> dict[str, str]:
    """Return {"title": ..., "description": ...} parsed from the agent answer.

    Strategy: prefer the last fenced ```json block; fall back to the last balanced
    top-level JSON object in the text. Missing fields become empty strings so the
    deterministic evaluators can fail them cleanly.
    """
    obj = _last_json_object(answer)
    title = ""
    description = ""
    if isinstance(obj, dict):
        title = _coerce_str(obj.get("title"))
        # Accept common aliases for the description field.
        description = _coerce_str(
            obj.get("description")
            if obj.get("description") is not None
            else obj.get("notes")
        )
    return {"title": title.strip(), "description": description.strip()}


def extract_trajectory(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Flatten the agent's tool calls into an ordered trajectory.

    Each entry: {"tool": <tool name>, "action": <ckan action>, "args": <dict>}.
    Tool names are the LangChain tool names (e.g. ``ckan_package_search``); the
    ``action`` strips the ``ckan_`` prefix for convenience.
    """
    trajectory: list[dict[str, Any]] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name", "")
            args = call.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            trajectory.append(
                {
                    "tool": name,
                    "action": name[len("ckan_"):] if name.startswith("ckan_") else name,
                    "args": args if isinstance(args, dict) else {},
                }
            )
    return trajectory


def collect_tool_outputs(messages: list[AnyMessage]) -> list[str]:
    """Return the text content of every ToolMessage, for grounding checks."""
    outputs: list[str] = []
    for message in messages:
        if message.__class__.__name__ == "ToolMessage":
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                outputs.append(content)
    return outputs


def _last_json_object(text: str) -> Any:
    if not text:
        return None
    blocks = _JSON_BLOCK.findall(text)
    for candidate in reversed(blocks):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # Fall back: scan for the last balanced {...} span and try to parse it.
    for span in reversed(_balanced_objects(text)):
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            continue
    return None


def _balanced_objects(text: str) -> list[str]:
    spans: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
    return spans


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
