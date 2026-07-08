from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.agents.ckan_registration.legacy_worker import LegacyCkanWorker
from app.agents.ckan_registration.schemas import (
    CkanAnalyzeInput,
    CkanApplyInput,
    CkanDryRunInput,
    CkanReviseInput,
    CkanShowInput,
)
from app.settings import Settings


TOOL_SPECS: dict[str, tuple[str, str, type[BaseModel], str]] = {
    "ckan_analyze": (
        "analyze",
        (
            "Build proposed CKAN dataset metadata and a resource plan from source URLs, staged files, "
            "or explicit dataset details. If a user mentions attachments/notebooks/scripts but no readable "
            "files or upload_dir are supplied, ask for those inputs instead of inferring metadata from clarification text."
        ),
        CkanAnalyzeInput,
        "safe-read",
    ),
    "ckan_revise": (
        "revise",
        "Revise a saved CKAN registration proposal without writing to CKAN.",
        CkanReviseInput,
        "safe-read",
    ),
    "ckan_dry_run": (
        "dry-run",
        "Compare the saved CKAN registration proposal against the target CKAN dataset without writing changes.",
        CkanDryRunInput,
        "safe-read",
    ),
    "ckan_apply": (
        "apply",
        "Create or patch the CKAN dataset and optionally upload resources. Requires approval exactly equal to REGISTER.",
        CkanApplyInput,
        "mutating",
    ),
    "ckan_show": (
        "show",
        "Return saved CKAN registration session state for debugging or review.",
        CkanShowInput,
        "safe-read",
    ),
}


def tool_model(tool_name: str) -> type[BaseModel]:
    try:
        return TOOL_SPECS[tool_name][2]
    except KeyError as exc:
        raise ValueError(f"Unknown CKAN registration tool: {tool_name}") from exc


def command_for_tool(tool_name: str) -> str:
    try:
        return TOOL_SPECS[tool_name][0]
    except KeyError as exc:
        raise ValueError(f"Unknown CKAN registration tool: {tool_name}") from exc


def _schema_for_model(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.setdefault("type", "object")
    return schema


def chat_completions_tool_schemas() -> list[dict[str, Any]]:
    tools = []
    for name, (_command, description, model, safety) in TOOL_SPECS.items():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{description} Safety: {safety}.",
                    "parameters": _schema_for_model(model),
                    "strict": False,
                },
            }
        )
    return tools


def responses_tool_schemas() -> list[dict[str, Any]]:
    tools = []
    for name, (_command, description, model, safety) in TOOL_SPECS.items():
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": f"{description} Safety: {safety}.",
                "parameters": _schema_for_model(model),
                "strict": False,
            }
        )
    return tools


def openai_tool_schema_bundle() -> dict[str, Any]:
    return {
        "name": "ckan-registration",
        "chat_completions_tools": chat_completions_tool_schemas(),
        "responses_tools": responses_tool_schemas(),
    }


class ToolExecutor:
    def __init__(self, settings: Settings) -> None:
        self.worker = LegacyCkanWorker(settings)

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        model = tool_model(tool_name)
        validated = model.model_validate(arguments).model_dump(mode="json", exclude_none=True)
        return self.worker.run(command_for_tool(tool_name), validated)
