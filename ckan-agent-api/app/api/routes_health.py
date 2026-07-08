from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health", operation_id="healthCheck")
def health() -> dict[str, object]:
    return {"ok": True, "service": "ckan-agent-api"}
