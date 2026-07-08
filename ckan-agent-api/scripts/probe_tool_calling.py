#!/usr/bin/env python3
"""R1 capability probe: does the configured LLM endpoint support OpenAI tool-calling?

Run it against the same endpoint/model the agent uses (read from app.settings, which
loads ckan-agent-api/.env). It makes 1-2 real chat requests with a tool definition and
reports whether the model returns a proper tool_call.

Usage:
    cd ckan-agent-api
    /Users/wmobley/opt/miniconda3/envs/ckan-agent-api/bin/python scripts/probe_tool_calling.py

Verdict:
    SUPPORTED (auto)   -> model volunteered a valid tool_call. Tool-calling personas are viable.
    SUPPORTED (forced) -> only complied when forced; usable but author prompts must nudge tool use.
    NOT SUPPORTED      -> model returned text instead of a tool_call. Keep tools MCP/evaluator-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import get_settings  # noqa: E402

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: the `openai` package is not installed in this environment.")
    raise SystemExit(2)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ckan_package_search",
            "description": "Search the CKAN catalog for datasets matching a free-text query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text search query."},
                    "rows": {"type": "integer", "description": "Max results.", "default": 10},
                },
                "required": ["query"],
            },
        },
    }
]

PROMPT = (
    "Check whether the CKAN catalog already has any datasets about the Yegua-Jackson "
    "groundwater aquifer before I register a new one."
)


def _client(settings):
    base_url = settings.openai_base_url or None
    return OpenAI(api_key=settings.openai_api_key, base_url=base_url)


def _call(client, settings, *, tool_choice):
    return client.chat.completions.create(
        model=settings.ckan_llm_model,
        messages=[
            {"role": "system", "content": "You can call tools. Use them when relevant."},
            {"role": "user", "content": PROMPT},
        ],
        tools=TOOLS,
        tool_choice=tool_choice,
        temperature=0.0,
        max_tokens=300,
        timeout=60,
    )


def _inspect(resp) -> tuple[bool, str]:
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        text = (msg.content or "")[:300]
        return False, f"no tool_calls; content: {text!r}"
    call = tool_calls[0]
    name = call.function.name
    raw_args = call.function.arguments
    try:
        args = json.loads(raw_args)
        ok = name == "ckan_package_search" and isinstance(args, dict) and "query" in args
        return ok, f"tool={name} args={args}"
    except json.JSONDecodeError:
        return False, f"tool={name} but arguments not valid JSON: {raw_args!r}"


def main() -> int:
    settings = get_settings()
    print(f"endpoint : {settings.openai_base_url or '(default openai)'}")
    print(f"model    : {settings.ckan_llm_model}")
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set in the environment/.env.")
        return 2
    client = _client(settings)

    print("\n[1] tool_choice='auto' (does the model volunteer a tool call?)")
    try:
        ok_auto, detail_auto = _inspect(_call(client, settings, tool_choice="auto"))
        print(f"    {'PASS' if ok_auto else 'no  '} - {detail_auto}")
    except Exception as exc:
        ok_auto = False
        print(f"    ERROR calling endpoint with tools: {exc}")
        print("    (If this is a 400/404, the endpoint may not accept the `tools` parameter at all.)")

    ok_forced = False
    if not ok_auto:
        print("\n[2] forced tool_choice (does it comply when required?)")
        forced = {"type": "function", "function": {"name": "ckan_package_search"}}
        try:
            ok_forced, detail_forced = _inspect(_call(client, settings, tool_choice=forced))
            print(f"    {'PASS' if ok_forced else 'no  '} - {detail_forced}")
        except Exception as exc:
            print(f"    ERROR with forced tool_choice: {exc}")

    print("\n==== VERDICT ====")
    if ok_auto:
        print("SUPPORTED (auto): tool-calling personas are viable. Proceed with the engine tool-loop.")
        return 0
    if ok_forced:
        print("SUPPORTED (forced only): usable, but prompts must nudge tool use; expect less reliability.")
        return 0
    print("NOT SUPPORTED: keep tools MCP/evaluator-only; do NOT rewrite the author into a tool loop.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
