from __future__ import annotations

import dataclasses

from starlette.requests import Request

import app.auth_context as ac
from app.agents.ckan_registration.auth import build_ckan_authorization_header
from app.settings import get_settings
from app.tools.handlers.ckan import _read_client


def _request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


def test_parse_authorization_variants():
    assert ac.parse_authorization(_request({"authorization": "Bearer abc.def.ghi"})) == "Bearer abc.def.ghi"
    assert ac.parse_authorization(_request({"authorization": "rawtoken"})) == "Bearer rawtoken"
    assert ac.parse_authorization(_request({})) is None


def test_bind_sets_and_resets():
    import asyncio

    async def run():
        gen = ac.bind_request_ckan_auth(_request({"authorization": "Bearer jwt-1"}))
        await gen.__anext__()
        assert ac.get_request_ckan_auth() == "Bearer jwt-1"
        await gen.aclose()
        assert ac.get_request_ckan_auth() is None

    asyncio.run(run())


def test_per_request_jwt_overrides_config():
    settings = dataclasses.replace(get_settings(), ckan_auth_mode="api_token", ckan_api_token="config-token")
    token = ac._ckan_auth.set("Bearer req-jwt")
    try:
        assert build_ckan_authorization_header(settings) == "Bearer req-jwt"
    finally:
        ac._ckan_auth.reset(token)


def test_falls_back_to_config_without_request_jwt():
    settings = dataclasses.replace(get_settings(), ckan_auth_mode="api_token", ckan_api_token="config-token")
    assert build_ckan_authorization_header(settings) == "config-token"


def test_read_client_uses_per_request_jwt():
    token = ac._ckan_auth.set("Bearer req-jwt")
    try:
        assert _read_client().authorization_header == "Bearer req-jwt"
    finally:
        ac._ckan_auth.reset(token)
