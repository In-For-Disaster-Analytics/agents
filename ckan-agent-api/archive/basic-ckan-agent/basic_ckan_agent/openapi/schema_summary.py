from __future__ import annotations

from typing import Any

from basic_ckan_agent.openapi.spec import resolve_ref


def summarize_schema(spec: dict[str, Any], schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        schema = resolve_ref(spec, schema["$ref"])

    required = schema.get("required", [])
    properties = schema.get("properties", {})
    lines: list[str] = []

    if required:
        lines.append(f"Required fields: {', '.join(required)}.")

    if properties:
        field_summaries = []
        for name, prop in list(properties.items())[:10]:
            if not isinstance(prop, dict):
                continue
            if "$ref" in prop:
                prop = resolve_ref(spec, prop["$ref"])

            typ = prop.get("type", "object")
            desc = prop.get("description", "")
            if desc:
                field_summaries.append(f"- {name} ({typ}): {desc[:180]}")
            else:
                field_summaries.append(f"- {name} ({typ})")

        if field_summaries:
            lines.append("Known fields:\n" + "\n".join(field_summaries))

    return "\n".join(lines)


def get_request_schema_for_operation(
    spec: dict[str, Any],
    operation: dict[str, Any],
) -> dict[str, Any]:
    request_body = operation.get("requestBody", {})
    content = request_body.get("content", {})

    for content_type in (
        "application/json",
        "application/x-www-form-urlencoded",
        "multipart/form-data",
    ):
        if content_type in content:
            return content[content_type].get("schema", {})

    return {}

