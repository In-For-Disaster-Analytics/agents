from __future__ import annotations

import json
from typing import Any

import requests

from basic_ckan_agent.ckan.client import call_ckan_action, redacted_headers
from basic_ckan_agent.ckan.constants import WRITE_ACTIONS
from basic_ckan_agent.ckan.response_compaction import compact_ckan_response
from basic_ckan_agent.logging_config import debug_print
from basic_ckan_agent.utils import safe_json_dumps


def execute_ckan_tool_call(
    *,
    tool_name: str,
    action_name: str,
    method: str,
    url: str,
    payload_json: str,
    approved: bool,
    write_approved: bool,
) -> str:
    debug_print(
        f"Tool called: {tool_name}",
        {
            "action_name": action_name,
            "method": method.upper(),
            "url": url,
            "approved": approved,
            "payload_json": payload_json,
        },
    )

    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError as exc:
        error = {
            "success": False,
            "action": action_name,
            "error": f"payload_json must be valid JSON: {exc}",
            "payload_json": payload_json,
        }
        debug_print("Payload JSON decode failed", error)
        return safe_json_dumps(error)

    if action_name in WRITE_ACTIONS and not (approved and write_approved):
        blocked = {
            "success": False,
            "blocked": True,
            "action": action_name,
            "proposed_payload": payload,
            "message": (
                f"{action_name} is a write action. No write was performed. "
                "Show the proposed_payload to the user for review. "
                'Apply it only after the current user message includes exact approval text "APPROVE WRITE".'
            ),
        }
        debug_print(f"Blocked write action: {action_name}", blocked)
        return safe_json_dumps(blocked)

    debug_print(
        f"Outgoing CKAN request: {action_name}",
        {
            "method": method.upper(),
            "url": url,
            "headers": redacted_headers(),
            "payload": payload,
        },
    )

    try:
        status_code, data = call_ckan_action(method=method, url=url, payload=payload)
    except requests.RequestException as exc:
        error = {
            "success": False,
            "action": action_name,
            "url": url,
            "error": str(exc),
        }
        debug_print("CKAN request exception", error)
        return safe_json_dumps(error)

    debug_print(
        f"CKAN response summary: {action_name}",
        _response_summary(action_name, url, status_code, data),
    )
    debug_print(f"Raw CKAN response: {action_name}", data)

    compact_data = compact_ckan_response(action_name, data)
    debug_print(
        f"Compact CKAN response returned to model: {action_name}",
        compact_data,
    )

    return safe_json_dumps(
        {
            "action": action_name,
            "url": url,
            "status_code": status_code,
            "response": compact_data,
        }
    )


def _response_summary(action_name: str, url: str, status_code: int, data: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "action": action_name,
        "url": url,
        "status_code": status_code,
    }

    if not isinstance(data, dict):
        return summary

    summary["ckan_success"] = data.get("success")
    summary["has_error"] = bool(data.get("error"))
    result = data.get("result")
    summary["result_type"] = type(result).__name__

    if isinstance(result, dict):
        summary["result_keys"] = list(result.keys())
        if "count" in result:
            summary["count"] = result.get("count")
        if "results" in result and isinstance(result["results"], list):
            summary["results_len"] = len(result["results"])
            if result["results"] and isinstance(result["results"][0], dict):
                first = result["results"][0]
                summary["first_result"] = {
                    "id": first.get("id"),
                    "name": first.get("name"),
                    "title": first.get("title"),
                }

    if data.get("error"):
        summary["error"] = data.get("error")

    return summary
