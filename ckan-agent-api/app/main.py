from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_agent import router as agent_router
from app.api.routes_auth import router as auth_router
from app.api.routes_health import router as health_router
from app.api.routes_schemas import router as schemas_router
from app.api.routes_uploads import router as uploads_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="CKAN Agent API",
        version="0.1.0",
        description="FastAPI and LangGraph service for CKAN registration agents.",
    )
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(schemas_router)
    app.include_router(agent_router)
    app.include_router(uploads_router)
    return app


app = create_app()
