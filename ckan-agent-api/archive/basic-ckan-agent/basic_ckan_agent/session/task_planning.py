from __future__ import annotations

import json
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage

WRITE_INTENT_WORDS = {"edit", "update", "modify", "change", "patch", "revise", "remove", "rename"}


def task_plan_prompt(user_question: str, selected_actions: list[str]) -> SystemMessage | None:
    action_set = set(selected_actions)
    has_write_action = bool(action_set & {"package_update", "package_patch"})
    has_write_intent = any(word in user_question.lower() for word in WRITE_INTENT_WORDS)

    if not has_write_action and not has_write_intent:
        return None

    return SystemMessage(
        content=(
            "This is a multi-step CKAN task. Work from this plan:\n"
            "1. Resolve the target dataset identity. If the user gave a display title or package_show returns "
            "Not Found, call package_search with the title or salient terms.\n"
            "2. Inspect the target dataset with package_show using a returned CKAN name or UUID.\n"
            "3. If the user has not specified exact metadata changes, ask for those changes before drafting a write.\n"
            "4. For edits, describe the intended package_patch or package_update payload before asking for approval.\n"
            "5. Do not call package_patch or package_update until the current user message includes "
            '"APPROVE WRITE".\n\n'
            "If a safe read-only next step is available, call the tool instead of saying that you will call it. "
            "In user-facing responses for incomplete tasks, include a concise Plan section with completed "
            "and next steps."
        )
    )


def file_metadata_plan_prompt(file_paths: list[str]) -> SystemMessage | None:
    if not file_paths:
        return None

    path_lines = "\n".join(f"- {path}" for path in file_paths)
    return SystemMessage(
        content=(
            "The current user request includes local file path(s):\n"
            f"{path_lines}\n\n"
            "Lead with the model's judgment, using local file tools as needed. For metadata planning from files:\n"
            "1. Inspect the path with file_stat before using content-specific tools.\n"
            "2. Choose the smallest useful file tool based on extension, MIME type, and the user's goal.\n"
            "3. Draft CKAN metadata only from user-provided context and file-tool evidence.\n"
            "4. For spatial files, use returned bbox or spatial_geojson evidence when proposing CKAN "
            "spatial metadata.\n"
            "5. Lead your answer with what you successfully recovered: summarize the package fields and resources "
            "you extracted, referring to each file by its name (for example resources.csv) rather than its full "
            "path. Only after that summary, ask concise follow-up questions for missing owner, organization, "
            "license, access, provenance, or temporal coverage.\n"
            '6. Do not call CKAN write tools unless the current user message includes "APPROVE WRITE".'
        )
    )


def package_show_404_recovery_call(
    messages: list[AnyMessage],
    selected_tool_names: list[str],
) -> AIMessage | None:
    if "ckan_package_search" not in selected_tool_names:
        return None

    payload = _last_tool_payload(messages)
    if payload.get("action") != "package_show" or payload.get("status_code") != 404:
        return None

    failed_id = _failed_package_show_id(messages)
    if not failed_id:
        return None

    return AIMessage(
        content=(
            "Plan:\n"
            "1. The package_show lookup failed, so resolve the dataset identity with package_search.\n"
            "2. Use a returned CKAN name or UUID for package_show.\n"
            "3. If this is an edit, ask for exact metadata changes and explicit approval before any write."
        ),
        tool_calls=[
            {
                "name": "ckan_package_search",
                "args": {
                    "payload_json": json.dumps({"q": failed_id, "rows": 5}),
                    "approved": False,
                },
                "id": f"call_recover_{uuid4().hex[:16]}",
                "type": "tool_call",
            }
        ],
    )


def _last_tool_payload(messages: list[AnyMessage]) -> dict:
    for message in reversed(messages):
        if message.__class__.__name__ != "ToolMessage":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else {}
    return {}


def _failed_package_show_id(messages: list[AnyMessage]) -> str:
    last_tool_id = _last_tool_call_id(messages)
    if not last_tool_id:
        return ""

    for message in reversed(messages):
        for call in getattr(message, "tool_calls", []) or []:
            if call.get("id") != last_tool_id:
                continue
            args = call.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return ""
            if not isinstance(args, dict):
                return ""

            payload_json = args.get("payload_json") or "{}"
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                return ""
            return str(payload.get("id") or "").strip()

    return ""


def _last_tool_call_id(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if message.__class__.__name__ == "ToolMessage":
            return str(getattr(message, "tool_call_id", "") or "")
    return ""
