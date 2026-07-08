"""Authentication endpoints — Tapis password-grant login proxy.

The browser cannot call the Tapis OAuth endpoint directly (CORS). This route proxies
the credential exchange server-side and returns the bare JWT so the UI can send it as
``Authorization: Bearer <jwt>`` on every subsequent request — which is exactly what
``auth_context.bind_request_ckan_auth`` already expects.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.agents.ckan_registration.auth import TapisAuth
from app.settings import Settings, get_settings

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    expires_in: int = 3600


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, settings: Settings = Depends(get_settings)) -> LoginResponse:
    """Exchange Tapis credentials for a JWT. The token is returned to the browser and
    sent back as ``Authorization: Bearer <jwt>`` on all agent and upload calls."""
    auth = TapisAuth(
        tapis_url=settings.ckan_tapis_url,
        username=body.username,
        password=body.password,
    )
    try:
        token = auth.access_token()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"Authentication failed: {exc}") from exc
    return LoginResponse(token=token, username=body.username)
