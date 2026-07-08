from __future__ import annotations

from fastapi import Request


SECRET_HEADER_NAMES = {
    "ckan-api-token",
    "x-ckan-api-token",
    "ckan-auth-mode",
    "x-ckan-auth-mode",
    "ckan-username",
    "x-ckan-username",
    "x-tapis-username",
    "ckan-password",
    "x-ckan-password",
    "x-tapis-password",
    "ckan-tapis-url",
    "x-ckan-tapis-url",
    "openai-api-key",
    "x-openai-api-key",
    "openai-base-url",
    "x-openai-base-url",
    "ckan-llm-model",
    "x-ckan-llm-model",
}


def normalize_header_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def extract_secret_headers(request: Request) -> dict[str, str]:
    out = {}
    for key, value in request.headers.items():
        normalized = normalize_header_name(key)
        if normalized in SECRET_HEADER_NAMES and value:
            out[normalized] = value.strip()
    return out


def merge_secret_headers(payload_headers: dict[str, str] | None, request: Request) -> dict[str, str] | None:
    headers = {**(payload_headers or {}), **extract_secret_headers(request)}
    return headers or None
