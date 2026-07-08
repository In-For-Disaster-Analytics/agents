from __future__ import annotations

from typing import Any


HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def operation_name(path: str, operation: dict[str, Any]) -> str:
    return operation.get("operationId") or path.strip("/").split("/")[-1]


def iter_operations(spec: dict[str, Any]):
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue

        for method, operation in path_item.items():
            method_lower = method.lower()
            if method_lower not in HTTP_METHODS or not isinstance(operation, dict):
                continue

            action_name = operation_name(path, operation)
            if action_name:
                yield path, method_lower, action_name, operation


def build_operation_catalog(
    spec: dict[str, Any],
    *,
    read_only_actions: set[str] | None = None,
    write_actions: set[str] | None = None,
) -> list[dict[str, Any]]:
    catalog = []
    for path, method, action_name, operation in iter_operations(spec):
        summary = operation.get("summary") or f"Call operation {action_name}"
        description = operation.get("description") or ""
        catalog.append(
            {
                "action": action_name,
                "method": method.upper(),
                "path": path,
                "summary": summary[:160],
                "description": description[:300],
                "read_only": action_name in (read_only_actions or set()),
                "write_action": action_name in (write_actions or set()),
            }
        )
    return catalog

