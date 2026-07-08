from __future__ import annotations

from copy import deepcopy

from fastapi import APIRouter, Request

from app.agents.ckan_registration.tools import openai_tool_schema_bundle
from app.settings import get_settings


router = APIRouter(tags=["schemas"])


@router.get("/schemas/openai-tools.json", operation_id="getOpenAIToolSchemas")
def get_openai_tool_schemas() -> dict[str, object]:
    return openai_tool_schema_bundle()


@router.get("/schemas/chatgpt-actions.json", operation_id="getChatGPTActionsSchema")
def get_chatgpt_actions_schema(request: Request) -> dict[str, object]:
    full_schema = deepcopy(request.app.openapi())
    allowed_prefixes = ("/v1/ckan-registration", "/health")
    full_schema["paths"] = {
        path: path_schema
        for path, path_schema in full_schema.get("paths", {}).items()
        if path.startswith(allowed_prefixes)
    }
    settings = get_settings()
    base_url = settings.public_base_url or str(request.base_url).rstrip("/")
    full_schema["servers"] = [{"url": base_url.rstrip("/")}]
    full_schema["info"]["title"] = "CKAN Registration Actions"
    full_schema["info"]["description"] = "Safe CKAN registration actions backed by FastAPI and LangGraph."
    return full_schema
