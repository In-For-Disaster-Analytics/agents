"""Shared OpenAI-compatible chat helper.

A single home for the chat call + JSON-response parsing used by the persona engine
(and, going forward, by the graph nodes — consolidation of the copy currently inside
``app/agents/ckan_registration/nodes.py`` is deferred to the graph-rewiring increment).

Kept free of ``Settings`` coupling so it is trivially testable and reusable: callers
pass ``model`` / ``api_key`` / ``base_url`` explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

try:  # preferred client
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Call throttle + retry (process-global; no Settings coupling).
#
# Every real round-trip — including the back-to-back calls inside the author's
# tool loop — goes through ``_gated_call``. It (1) spaces calls by at least
# ``_min_interval_seconds`` so we stop hammering a rate-limited model group,
# (2) emits a per-call timing log so the cadence is visible, and (3) retries on
# 429 with backoff (honoring ``Retry-After``) so one throttle hiccup no longer
# kills a whole drafting round. Configure via ``configure_throttle`` (called by
# ``app.settings.get_settings`` at startup) or the ``LLM_CALL_DELAY_SECONDS`` /
# ``LLM_MAX_RETRIES`` env vars.
# ---------------------------------------------------------------------------
_min_interval_seconds: float = float(os.getenv("LLM_CALL_DELAY_SECONDS") or 0.0)
_max_retries: int = int(os.getenv("LLM_MAX_RETRIES") or 4)
_last_call_monotonic: float | None = None


def configure_throttle(*, delay_seconds: float | None = None, max_retries: int | None = None) -> None:
    """Set the inter-call delay and 429-retry budget (idempotent; safe to call again)."""
    global _min_interval_seconds, _max_retries
    if delay_seconds is not None:
        _min_interval_seconds = max(0.0, float(delay_seconds))
    if max_retries is not None:
        _max_retries = max(0, int(max_retries))


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "ratelimit" in text


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _gated_call(fn: Callable[[], Any], *, model: str, kind: str = "chat") -> Any:
    """Run ``fn`` behind the shared throttle: space calls, time them, retry on 429."""
    global _last_call_monotonic
    attempt = 0
    while True:
        if _min_interval_seconds > 0 and _last_call_monotonic is not None:
            wait = _min_interval_seconds - (time.monotonic() - _last_call_monotonic)
            if wait > 0:
                logger.debug("⏳ LLM throttle: sleeping %.2fs before next %s call", wait, kind)
                time.sleep(wait)
        started = time.monotonic()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - classify then re-raise or retry
            _last_call_monotonic = time.monotonic()
            elapsed = _last_call_monotonic - started
            if _is_rate_limit(exc) and attempt < _max_retries:
                attempt += 1
                backoff = _retry_after_seconds(exc)
                if backoff is None:
                    backoff = min(60.0, (2.0 ** (attempt - 1)) * max(1.0, _min_interval_seconds or 1.0))
                logger.warning(
                    "⏱ LLM %s model=%s rate-limited after %.2fs; retry %d/%d in %.1fs",
                    kind, model, elapsed, attempt, _max_retries, backoff,
                )
                time.sleep(backoff)
                continue
            logger.info("⏱ LLM %s model=%s FAILED after %.2fs: %s", kind, model, elapsed, exc)
            raise
        _last_call_monotonic = time.monotonic()
        logger.info("⏱ LLM %s model=%s ok in %.2fs (attempt %d)", kind, model, _last_call_monotonic - started, attempt + 1)
        return result


def invoke_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str = "",
    temperature: float = 0.1,
    max_tokens: int = 3000,
    timeout: int = 60,
) -> str:
    """Send chat messages and return the assistant's text content."""
    def _do() -> str:
        if ChatOpenAI is not None:
            kwargs: dict[str, Any] = {
                "model": model,
                "api_key": api_key,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if base_url:
                kwargs["base_url"] = base_url
            response = ChatOpenAI(**kwargs).invoke(
                [(_role(m["role"]), m["content"]) for m in messages]
            )
            content = getattr(response, "content", response)
            return _content_text(content)

        if OpenAI is not None:  # pragma: no cover - exercised only without langchain
            client = OpenAI(api_key=api_key, base_url=base_url or None)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content or ""

        import requests  # pragma: no cover - last-resort fallback

        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=timeout,
        )
        resp.raise_for_status()
        choices = resp.json().get("choices") or []
        return ((choices[0] if choices else {}).get("message") or {}).get("content") or ""

    return _gated_call(_do, model=model, kind="chat")


def invoke_chat_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model: str,
    api_key: str,
    base_url: str = "",
    temperature: float = 0.1,
    max_tokens: int = 1500,
    timeout: int = 90,
) -> dict[str, Any]:
    """One tool-calling chat turn (OpenAI tools API).

    Returns ``{"content": str|None, "tool_calls": [{"id","name","arguments"(dict)}],
    "raw_message": <assistant message dict>}``. ``raw_message`` is appended verbatim to
    the message list on the next turn so the tool-call protocol round-trips correctly.
    """
    if OpenAI is None:  # pragma: no cover - openai is a declared dependency
        raise RuntimeError("openai package is required for tool-calling chat.")
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    message = _gated_call(
        lambda: client.chat.completions.create(**kwargs), model=model, kind="chat+tools"
    ).choices[0].message

    tool_calls: list[dict[str, Any]] = []
    for call in getattr(message, "tool_calls", None) or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        tool_calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})

    raw = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else {
        "role": "assistant",
        "content": message.content,
    }
    return {"content": message.content, "tool_calls": tool_calls, "raw_message": raw}


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating ```json fences and prose.

    Returns ``{}`` when no JSON object can be recovered (callers decide how to treat it).
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _role(role: str) -> str:
    return {"assistant": "ai", "system": "system", "user": "human"}.get(role, role)


def _content_text(content: Any) -> str:
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
