from __future__ import annotations

import ast
import json

from langchain_core.messages import AnyMessage


def model_bad_request_recovery_message(error: Exception, messages: list[AnyMessage]) -> str:
    provider_output = _provider_model_output(error)
    if provider_output:
        return f"{provider_output}\n\nNo write action was performed."

    tool_payload = _last_tool_payload(messages)
    action = tool_payload.get("action")
    status_code = tool_payload.get("status_code")
    response = tool_payload.get("response") if isinstance(tool_payload.get("response"), dict) else {}

    if action == "package_show" and status_code == 404:
        return (
            "CKAN returned Not Found for `package_show`. The value passed was not a CKAN dataset "
            "`name` or UUID. If the user gave a display title, search for that title with "
            "`package_search`, then use the returned `name` or `id` for `package_show`, "
            "`package_patch`, or `package_update`. No write action was performed."
        )

    if response.get("error"):
        return f"CKAN returned an error for `{action}`: {response['error']}. No write action was performed."

    return (
        "The model provider rejected a function-calling response while handling the last tool result. "
        f"No write action was performed. Provider error: {error}"
    )


def _provider_model_output(error: Exception) -> str:
    text = str(error)
    try:
        outer = ast.literal_eval(text.split(" - ", 1)[1])
    except (IndexError, SyntaxError, ValueError):
        return ""

    if not isinstance(outer, dict):
        return ""
    outer_error = outer.get("error")
    if not isinstance(outer_error, dict):
        return ""
    message = str(outer_error.get("message") or "")

    marker = "SambanovaException - Error code: 400 - "
    if marker not in message:
        return ""

    start = message.find("{", message.find(marker))
    nested_text = _balanced_braced_text(message, start)
    if not nested_text:
        return ""

    try:
        nested = ast.literal_eval(nested_text)
    except (SyntaxError, ValueError):
        return ""

    if not isinstance(nested, dict):
        return ""
    output = nested.get("error_model_output")
    return str(output or "").strip()


def _balanced_braced_text(text: str, start: int) -> str:
    if start < 0 or start >= len(text) or text[start] != "{":
        return ""

    depth = 0
    quote = ""
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return ""


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

