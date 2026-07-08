from __future__ import annotations

from typing import Any

from basic_ckan_agent.settings import ckan_base_url


def get_ckan_action_url(spec: dict[str, Any], path: str) -> str:
    servers = spec.get("servers") or [{"url": "/api/3/action"}]
    server_url = servers[0].get("url", "/api/3/action")

    if server_url.startswith("http://") or server_url.startswith("https://"):
        base = server_url.rstrip("/")
    else:
        base = f"{ckan_base_url()}/{server_url.lstrip('/')}".rstrip("/")

    return f"{base}/{path.lstrip('/')}"

