from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import requests

from app.settings import Settings


def clean_text(value: object) -> str:
    return str(value or "").strip()


@dataclass
class TapisTokenCache:
    token: str = ""
    expires_at_monotonic: float = 0.0

    def get(self) -> str:
        if self.token and monotonic() < self.expires_at_monotonic:
            return self.token
        return ""

    def set(self, token: str, ttl_seconds: int = 3300) -> None:
        self.token = token
        self.expires_at_monotonic = monotonic() + ttl_seconds


class TapisAuth:
    def __init__(self, *, tapis_url: str, username: str, password: str, timeout: int = 30) -> None:
        self.tapis_url = tapis_url
        self.username = username
        self.password = password
        self.timeout = timeout
        self._cache = TapisTokenCache()

    def access_token(self, *, force_refresh: bool = False) -> str:
        cached = "" if force_refresh else self._cache.get()
        if cached:
            return cached
        if not clean_text(self.username) or not self.password:
            raise ValueError("CKAN_USERNAME and CKAN_PASSWORD are required for Tapis password authentication.")
        response = requests.post(
            self.tapis_url,
            data={"username": self.username, "password": self.password, "grant_type": "password"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload["result"]["access_token"]["access_token"]
        self._cache.set(token)
        return token

    def ckan_authorization_header(self) -> str:
        return f"Bearer {self.access_token()}"


def build_ckan_authorization_header(settings: Settings, *, required: bool = False) -> str | None:
    # A per-request JWT (from the conversation's Authorization header) takes precedence over
    # any server-configured auth mode, so a chat user authenticates to CKAN as themselves.
    from app.auth_context import get_request_ckan_auth

    per_request = get_request_ckan_auth()
    if per_request:
        return per_request

    mode = clean_text(settings.ckan_auth_mode).lower() or "tapis_password"
    if mode == "api_token":
        token = clean_text(settings.ckan_api_token)
        if token:
            return token
        if required:
            raise ValueError("CKAN_API_TOKEN is required when CKAN_AUTH_MODE=api_token.")
        return None
    if mode == "tapis_password":
        auth = TapisAuth(
            tapis_url=settings.ckan_tapis_url,
            username=settings.ckan_username,
            password=settings.ckan_password,
        )
        if required:
            return auth.ckan_authorization_header()
        if not clean_text(settings.ckan_username) or not settings.ckan_password:
            return None
        return auth.ckan_authorization_header()
    raise ValueError(f"Unsupported CKAN_AUTH_MODE: {settings.ckan_auth_mode}")
