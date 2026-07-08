"""Structured tool-result envelope (revived from the archived basic-ckan-agent)."""

from __future__ import annotations

from typing import Any


def tool_success(tool: str, result: Any) -> dict[str, Any]:
    return {"success": True, "tool": tool, "result": result}


def tool_error(tool: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    error.update(extra)
    return {"success": False, "tool": tool, "error": error}
