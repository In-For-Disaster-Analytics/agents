from __future__ import annotations

from typing import Any

import requests

from basic_ckan_agent.settings import ckan_api_token


def ckan_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # CKAN usually expects the raw token in Authorization, not Bearer <token>.
    if ckan_api_token():
        headers["Authorization"] = ckan_api_token()

    return headers


def redacted_headers() -> dict[str, str]:
    return {
        key: value if key.lower() != "authorization" else "***REDACTED***"
        for key, value in ckan_headers().items()
    }


def call_ckan_action(*, method: str, url: str, payload: dict[str, Any]) -> tuple[int, Any]:
    if method.lower() == "get":
        response = requests.get(url, headers=ckan_headers(), params=payload, timeout=30)
    else:
        response = requests.request(method.upper(), url, headers=ckan_headers(), json=payload, timeout=30)

    try:
        data = response.json()
    except ValueError:
        data = {
            "success": False,
            "status_code": response.status_code,
            "text": response.text[:3000],
        }

    return response.status_code, data

