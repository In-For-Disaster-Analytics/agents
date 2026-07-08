"""The LangSmith target: run the LangGraph agent under a prompt/model config.

``build_eval_target`` returns a callable that LangSmith invokes per dataset
example. It mirrors the ``graph.with_config(configurable={...})`` pattern from the
spec: each target binds one (system_prompt, model) pair, runs the real agent, and
returns the generated metadata plus the tool trajectory for the evaluators.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from basic_ckan_agent.evaluation.extraction import (
    collect_tool_outputs,
    extract_metadata,
    extract_trajectory,
)
from basic_ckan_agent.logging_config import logger
from basic_ckan_agent.runtime.graph import ChatSession

# Appended to metadata-generation requests so the agent emits parseable output.
_JSON_INSTRUCTION = (
    "When you have drafted the metadata, end your reply with a single fenced ```json "
    'code block containing an object with exactly the keys "title" and "description". '
    "Base both fields only on the source metadata and tool outputs; do not invent facts."
)


def build_eval_target(
    prompt_text: str | None,
    model_name: str | None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a LangSmith target bound to one prompt/model configuration."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        request = build_request(inputs)
        session = ChatSession(system_prompt=prompt_text, model_name=model_name)
        try:
            result = session.ask_with_trace(request)
            answer = result.answer
            messages = result.messages
        except Exception as exc:  # keep the experiment going on a single failure
            logger.exception("Eval target failed for inputs=%s", inputs)
            return {
                "title": "",
                "description": "",
                "answer": f"ERROR: {exc}",
                "trajectory": [],
                "tool_outputs": [],
                "error": str(exc),
            }

        metadata = extract_metadata(answer)
        trajectory = extract_trajectory(messages)
        return {
            "title": metadata["title"],
            "description": metadata["description"],
            "answer": answer,
            "trajectory": trajectory,
            "tools_called": [step["action"] for step in trajectory],
            "tool_outputs": collect_tool_outputs(messages),
        }

    return target


def build_request(inputs: dict[str, Any]) -> str:
    """Compose a single user message from a dataset example's input fields.

    Supported input keys: ``question`` (free-text request), ``metadata`` (source
    metadata dict), ``source_context`` (extra text), and ``task_type`` (one of
    ``metadata``, ``search``, ``resources``). Metadata tasks get the JSON-output
    instruction appended.
    """
    parts: list[str] = []
    question = inputs.get("question")
    metadata = inputs.get("metadata")
    source_context = inputs.get("source_context")
    task_type = (inputs.get("task_type") or _infer_task_type(inputs)).lower()

    if question:
        parts.append(str(question))
    elif task_type == "metadata":
        parts.append("Generate a CKAN dataset title and description from the source metadata below.")

    if metadata:
        parts.append("Source metadata:\n" + json.dumps(metadata, indent=2, ensure_ascii=False))
    if source_context:
        parts.append("Source/tool context:\n" + str(source_context))

    if task_type == "metadata":
        parts.append(_JSON_INSTRUCTION)

    return "\n\n".join(parts)


def _infer_task_type(inputs: dict[str, Any]) -> str:
    if inputs.get("metadata") and not inputs.get("question"):
        return "metadata"
    question = str(inputs.get("question") or "").lower()
    if any(word in question for word in ["title", "description", "metadata", "describe"]):
        return "metadata"
    if any(word in question for word in ["resource", "file", "download"]):
        return "resources"
    return "search"
