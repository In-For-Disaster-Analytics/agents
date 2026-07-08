from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import AnyMessage, SystemMessage

from basic_ckan_agent.ckan.constants import WRITE_ACTIONS


@dataclass(frozen=True)
class ActiveDataset:
    id: str
    name: str
    title: str


@dataclass(frozen=True)
class PendingWrite:
    action: str
    payload: dict


def memory_prompt(messages: list[AnyMessage]) -> SystemMessage | None:
    dataset = active_dataset(messages)
    pending = pending_write(messages)

    if not dataset and not pending:
        return None

    lines = ["Conversation memory:"]
    if dataset:
        lines.extend(
            [
                f"- Active CKAN dataset title: {dataset.title}",
                f"- Active CKAN dataset name: {dataset.name}",
                f"- Active CKAN dataset id: {dataset.id}",
            ]
        )
    if pending:
        lines.extend(
            [
                f"- Pending write action: {pending.action}",
                f"- Pending write payload: {json.dumps(pending.payload, ensure_ascii=False)}",
            ]
        )

    return SystemMessage(
        content=(
            "\n".join(lines)
            + "\n\n"
            "Use this active dataset for follow-up references like 'this one', 'the title', "
            "or 'remove text from the title'. Do not rediscover it unless the user changes targets. "
            'If there is a pending write and the current user says "APPROVE WRITE", call the pending write action '
            "with the pending payload."
        )
    )


def active_dataset(messages: list[AnyMessage]) -> ActiveDataset | None:
    for message in reversed(messages):
        payload = _tool_payload(message)
        if payload.get("action") != "package_show" or payload.get("status_code") != 200:
            continue

        response = payload.get("response")
        if not isinstance(response, dict):
            continue
        result = response.get("result")
        if not isinstance(result, dict):
            continue

        dataset_id = str(result.get("id") or "").strip()
        name = str(result.get("name") or "").strip()
        title = str(result.get("title") or "").strip()
        if dataset_id and name and title:
            return ActiveDataset(id=dataset_id, name=name, title=title)

    return None


def pending_write(messages: list[AnyMessage]) -> PendingWrite | None:
    for message in reversed(messages):
        payload = _tool_payload(message)
        action = str(payload.get("action") or "")
        proposed_payload = payload.get("proposed_payload")
        if action in WRITE_ACTIONS and payload.get("blocked") and isinstance(proposed_payload, dict):
            return PendingWrite(action=action, payload=proposed_payload)

        recovered = _pending_write_from_ai_text(message)
        if recovered:
            return recovered
    return None


def _tool_payload(message: AnyMessage) -> dict:
    if message.__class__.__name__ != "ToolMessage":
        return {}

    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return {}

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _pending_write_from_ai_text(message: AnyMessage) -> PendingWrite | None:
    if message.__class__.__name__ != "AIMessage":
        return None

    content = getattr(message, "content", "")
    if not isinstance(content, str) or "APPROVE WRITE" not in content:
        return None

    payload = _json_fence_payload(content)
    if not payload:
        return None

    action = "package_patch" if "package_patch" in content else "package_update"
    return PendingWrite(action=action, payload=payload)


def _json_fence_payload(text: str) -> dict:
    marker = "```json"
    start = text.find(marker)
    if start < 0:
        return {}
    start += len(marker)
    end = text.find("```", start)
    if end < 0:
        return {}

    try:
        payload = json.loads(text[start:end].strip())
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}

