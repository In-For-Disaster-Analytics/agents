from __future__ import annotations

from typing import Any

import requests

from basic_ckan_agent.logging_config import debug_print


def load_openapi_schema(url: str) -> dict[str, Any]:
    debug_print("Fetching OpenAPI schema from URL", url)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        spec = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch OpenAPI schema from {url}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"OpenAPI schema response was not valid JSON: {exc}") from exc

    debug_print(
        "Loaded OpenAPI schema summary",
        {
            "openapi": spec.get("openapi"),
            "title": spec.get("info", {}).get("title"),
            "version": spec.get("info", {}).get("version"),
            "servers": spec.get("servers"),
            "path_count": len(spec.get("paths", {})),
            "first_15_paths": list(spec.get("paths", {}).keys())[:15],
        },
    )
    return spec


def resolve_ref(spec: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        return {}

    cur: Any = spec
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(part, {})

    return cur if isinstance(cur, dict) else {}
