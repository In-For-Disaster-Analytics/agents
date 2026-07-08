from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CATALOG_PATH = Path(__file__).with_name("tool_catalog.yaml")


@lru_cache
def load_file_tool_catalog() -> list[dict[str, Any]]:
    raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    tools = raw.get("tools", [])
    return tools if isinstance(tools, list) else []


def build_file_tool_catalog(enabled_names: set[str] | None = None) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for item in load_file_tool_catalog():
        name = item.get("name")
        if not isinstance(name, str):
            continue
        if enabled_names is not None and name not in enabled_names:
            continue
        catalog.append(
            {
                "tool": name,
                "summary": str(item.get("summary") or item.get("description") or "")[:180],
                "description": str(item.get("description") or "")[:400],
                "read_only": True,
                "use_when": item.get("use_when", []),
                "limitations": item.get("limitations", []),
            }
        )
    return catalog


def description_for_tool(name: str, fallback: str) -> str:
    for item in load_file_tool_catalog():
        if item.get("name") != name:
            continue
        description = str(item.get("description") or fallback).strip()
        use_when = item.get("use_when") or []
        returns = item.get("returns") or {}
        safety = item.get("safety") or []
        parts = [description]
        if use_when:
            parts.append("Use when:\n" + "\n".join(f"- {entry}" for entry in use_when))
        if returns:
            parts.append(f"Returns: {returns}")
        if safety:
            parts.append("Safety:\n" + "\n".join(f"- {entry}" for entry in safety))
        return "\n\n".join(parts)
    return fallback
