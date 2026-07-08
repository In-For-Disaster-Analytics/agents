"""Read-only CKAN tool handlers.

Thin wrappers over ``CkanClient`` (spec R4: read + dry-run only — no create/update/apply
here; writes stay in the gated graph path). Each handler takes a validated ``args`` dict.

Auth honors ``CKAN_AUTH_MODE`` like the rest of the app: ``tapis_password`` (default) mints a
Tapis Bearer token from ``CKAN_USERNAME``/``CKAN_PASSWORD``; ``api_token`` uses ``CKAN_API_TOKEN``.
Falls back to anonymous (public datasets) if no credentials are configured.
"""

from __future__ import annotations

from typing import Any

from app.agents.ckan_registration.auth import build_ckan_authorization_header
from app.agents.ckan_registration.ckan_client import CkanClient
from app.settings import get_settings


def _read_client() -> CkanClient:
    settings = get_settings()
    try:
        auth_header = build_ckan_authorization_header(settings, required=False)
    except Exception:
        auth_header = None  # read tools degrade to anonymous rather than failing the call
    return CkanClient(base_url=settings.ckan_url, authorization_header=auth_header, timeout=30)


def package_show(args: dict[str, Any]) -> Any:
    return _read_client().package_show(str(args["dataset_name"]))


def package_search(args: dict[str, Any]) -> Any:
    return _read_client().package_search(str(args["query"]), rows=int(args.get("rows", 10)))


def organization_list(args: dict[str, Any]) -> Any:
    return _read_client().organization_list()


def resolve_org(args: dict[str, Any]) -> Any:
    return _read_client().resolve_organization_id(str(args["organization"]))


def dry_run_diff(args: dict[str, Any]) -> dict[str, Any]:
    """Compare a desired field set against the live CKAN package (read-only diff)."""
    desired = args.get("desired") or {}
    if not isinstance(desired, dict):
        desired = {}
    existing = _read_client().package_show(str(args["dataset_name"]))
    if existing is None:
        return {
            "existing_found": False,
            "changes": [{"field": k, "from": None, "to": v} for k, v in desired.items()],
        }
    changes = [
        {"field": k, "from": existing.get(k), "to": v}
        for k, v in desired.items()
        if existing.get(k) != v
    ]
    return {"existing_found": True, "changes": changes}
