from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from basic_ckan_agent.openapi.schema_summary import get_request_schema_for_operation, summarize_schema


@dataclass(frozen=True)
class OperationToolSpec:
    action_name: str
    method: str
    path: str
    summary: str
    description: str
    schema_summary: str


def operation_tool_spec(
    *,
    spec: dict[str, Any],
    action_name: str,
    method: str,
    path: str,
    operation: dict[str, Any],
) -> OperationToolSpec:
    request_schema = get_request_schema_for_operation(spec, operation)
    return OperationToolSpec(
        action_name=action_name,
        method=method,
        path=path,
        summary=operation.get("summary") or f"Call operation {action_name}",
        description=operation.get("description") or "",
        schema_summary=summarize_schema(spec, request_schema),
    )

