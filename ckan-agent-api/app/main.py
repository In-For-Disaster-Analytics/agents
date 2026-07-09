from __future__ import annotations

import logging
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_agent import router as agent_router
from app.api.routes_auth import router as auth_router
from app.api.routes_health import router as health_router
from app.api.routes_schemas import router as schemas_router
from app.api.routes_uploads import router as uploads_router

logger = logging.getLogger(__name__)

_UPLOAD_TTL_HOURS = 24
_SWEEP_INTERVAL_SECONDS = 3600  # check hourly


def _upload_sweep_loop(upload_root: Path, ttl_seconds: float) -> None:
    """Daemon thread: delete upload directories older than ttl_seconds."""
    while True:
        time.sleep(_SWEEP_INTERVAL_SECONDS)
        if not upload_root.is_dir():
            continue
        cutoff = time.time() - ttl_seconds
        for entry in upload_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    logger.info("TTL sweep: removed upload dir %s", entry.name)
            except OSError:
                pass


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from app.settings import get_settings
    settings = get_settings()
    t = threading.Thread(
        target=_upload_sweep_loop,
        args=(settings.upload_root, _UPLOAD_TTL_HOURS * 3600),
        name="upload-ttl-sweep",
        daemon=True,
    )
    t.start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="CKAN Agent API",
        version="0.1.0",
        description="FastAPI and LangGraph service for CKAN registration agents.",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(schemas_router)
    app.include_router(agent_router)
    app.include_router(uploads_router)
    return app


app = create_app()
