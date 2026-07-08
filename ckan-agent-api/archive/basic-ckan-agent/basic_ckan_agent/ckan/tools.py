from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from basic_ckan_agent.ckan.action_execution import execute_ckan_tool_call
from basic_ckan_agent.ckan.examples import COMMON_PAYLOAD_EXAMPLES
from basic_ckan_agent.ckan.urls import get_ckan_action_url
from basic_ckan_agent.logging_config import debug_print
from basic_ckan_agent.openapi.catalog import iter_operations
from basic_ckan_agent.openapi.tools import operation_tool_spec


class CkanActionInput(BaseModel):
    payload_json: str = Field(
        default="{}",
        description=(
            "JSON object string to send to the CKAN action endpoint. "
            'Example: {"q":"flood","rows":5} for package_search, '
            'or {"id":"dataset-name"} for package_show.'
        ),
    )
    approved: bool = Field(
        default=False,
        description=(
            "Set true only when the user explicitly approved a write action. "
            "Read-only actions ignore this."
        ),
    )


def _tool_description(
    *,
    summary: str,
    action_name: str,
    method: str,
    description: str,
    schema_summary: str,
) -> str:
    request_guidance = schema_summary or "Pass a JSON object string matching the CKAN action input."
    return f"""
{summary}

CKAN action: {action_name}
HTTP method: {method.upper()}

{description[:500]}

Request body guidance:
{request_guidance}

The tool accepts:
- payload_json: JSON string body to send to CKAN.
- approved: boolean, only needed for write actions.

{COMMON_PAYLOAD_EXAMPLES}
""".strip()


def make_ckan_tool(
    *,
    spec: dict[str, Any],
    action_name: str,
    method: str,
    path: str,
    operation: dict[str, Any],
    write_approved: bool = False,
) -> StructuredTool:
    tool_name = f"ckan_{action_name}".replace("-", "_")
    url = get_ckan_action_url(spec, path)
    tool_spec = operation_tool_spec(
        spec=spec,
        action_name=action_name,
        method=method,
        path=path,
        operation=operation,
    )

    full_description = _tool_description(
        summary=tool_spec.summary,
        action_name=action_name,
        method=method,
        description=tool_spec.description,
        schema_summary=tool_spec.schema_summary,
    )

    def run_ckan_action(payload_json: str = "{}", approved: bool = False) -> str:
        return execute_ckan_tool_call(
            tool_name=tool_name,
            action_name=action_name,
            method=method,
            url=url,
            payload_json=payload_json,
            approved=approved,
            write_approved=write_approved,
        )

    return StructuredTool.from_function(
        func=run_ckan_action,
        name=tool_name,
        description=full_description,
        args_schema=CkanActionInput,
    )


def build_tools_from_openapi(
    spec: dict[str, Any],
    allowed_actions: set[str] | None = None,
    write_approved: bool = False,
) -> list[StructuredTool]:
    tools: list[StructuredTool] = []

    for path, method, action_name, operation in iter_operations(spec):
        if allowed_actions is not None and action_name not in allowed_actions:
            continue

        tools.append(
            make_ckan_tool(
                spec=spec,
                action_name=action_name,
                method=method,
                path=path,
                operation=operation,
                write_approved=write_approved,
            )
        )

    debug_print(
        "Generated selected CKAN tools",
        {
            "count": len(tools),
            "tools": [tool.name for tool in tools],
        },
    )
    return tools
