from __future__ import annotations

from typing import Any


def tool_success(tool: str, result: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": True,
        "tool": tool,
        "result": result,
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def tool_error(tool: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "tool": tool,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if extra:
        payload["error"].update(extra)
    return payload
