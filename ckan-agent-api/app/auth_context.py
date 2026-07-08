"""Per-request CKAN authorization (no storage).

A conversation supplies its CKAN credential as the HTTP ``Authorization: Bearer <jwt>``
header (the chat client's API-key field). We capture it per request into a contextvar and
use it as the CKAN ``Authorization`` header for that turn's CKAN calls — both the read tools
and the legacy dry-run/apply path. Nothing is persisted; the client re-sends the header each
turn, so it naturally covers the whole conversation without a secret at rest.
"""

from __future__ import annotations

import contextvars
from collections.abc import AsyncIterator

from fastapi import Request

_ckan_auth: contextvars.ContextVar[str | None] = contextvars.ContextVar("ckan_request_auth", default=None)


def get_request_ckan_auth() -> str | None:
    """The per-request CKAN Authorization header value (e.g. ``Bearer <jwt>``), or None."""
    return _ckan_auth.get()


def parse_authorization(request: Request) -> str | None:
    """Read the incoming Authorization header into a CKAN-ready header value.

    A value with a scheme (``Bearer <jwt>``) is used verbatim. A bare token is presented as a
    Bearer token (Tapis-fronted CKAN expects ``Bearer <jwt>``). Empty → None (fall back to
    server-configured auth).
    """
    value = (request.headers.get("authorization") or "").strip()
    if not value:
        return None
    return value if " " in value else f"Bearer {value}"


async def bind_request_ckan_auth(request: Request) -> AsyncIterator[None]:
    """FastAPI dependency: bind the request's Authorization as the CKAN auth for this call.

    Async so set/reset run in the same event-loop context (a sync yield-dependency would
    reset the contextvar Token in a different thread/context and raise). The sync path
    operation runs in a threadpool with a copy of this context, so the value still propagates.
    """
    token = _ckan_auth.set(parse_authorization(request))
    try:
        yield
    finally:
        _ckan_auth.reset(token)
