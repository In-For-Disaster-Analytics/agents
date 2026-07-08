"""Ground ``owner_org`` in the live CKAN organization list (spec 2026-06-30).

The deployment configures a default ``CKAN_OWNER_ORG``, but that label may not exist on the
target portal (e.g. the default is ``DSO-Institute`` while a local portal only has ``twdb-gam``).
Rather than pass an invalid org through to the write, we resolve it against the portal's real
organizations at persona-seed time:

- configured org matches a real org  → use the canonical org name
- configured doesn't match, exactly one org exists → use that one
- configured doesn't match, multiple orgs exist → ambiguous (caller asks the user to choose)
- no orgs / portal unreachable → leave the configured default (best-effort; never blocks)
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.ckan_registration.ckan_client import CkanClient
from app.settings import Settings

logger = logging.getLogger(__name__)


def _labels(org: dict[str, Any]) -> set[str]:
    return {
        str(org.get(k) or "").strip().casefold()
        for k in ("id", "name", "title", "display_name")
    } - {""}


def resolve_owner_org_choice(
    orgs: list[dict[str, Any]], configured: str
) -> tuple[str | None, bool, list[dict[str, str]]]:
    """Resolve a configured org label against live orgs.

    Returns ``(resolved_name | None, ambiguous, options)``:
    - ``resolved_name`` is the canonical org ``name`` to use (None when it can't be decided).
    - ``ambiguous`` is True when the user must choose (multiple orgs, no configured match).
    - ``options`` is the choice list (``{name, title}``) when ambiguous.
    """
    configured_norm = str(configured or "").strip().casefold()
    if configured_norm:
        for org in orgs:
            if configured_norm in _labels(org):
                return (str(org.get("name") or org.get("id") or configured), False, [])
    if len(orgs) == 1:
        only = orgs[0]
        return (str(only.get("name") or only.get("id") or ""), False, [])
    if not orgs:
        return (None, False, [])
    options = [
        {"name": str(o.get("name") or ""), "title": str(o.get("title") or o.get("display_name") or "")}
        for o in orgs
        if o.get("name") or o.get("id")
    ]
    return (None, True, options)


def _read_client(settings: Settings) -> CkanClient:
    try:
        from app.agents.ckan_registration.auth import build_ckan_authorization_header

        auth_header = build_ckan_authorization_header(settings, required=False)
    except Exception:  # noqa: BLE001 - reads degrade to anonymous rather than failing
        auth_header = None
    return CkanClient(base_url=settings.ckan_url, authorization_header=auth_header, timeout=30)


def fetch_orgs(settings: Settings) -> list[dict[str, Any]]:
    """Best-effort fetch of the portal's organizations (read-only; degrades to anonymous)."""
    return _read_client(settings).organization_list()


def _normalize_license(value: str) -> str:
    return str(value or "").strip().casefold().replace(" ", "-").replace("_", "-")


def resolve_license_id(licenses: list[dict[str, Any]], configured: str) -> str | None:
    """Resolve a configured ``license_id`` against the portal's enabled licenses.

    Returns the canonical license ``id`` when a match is found (exact id, then normalized id/title),
    else None (caller keeps the configured value). Licenses have no sensible single-fallback, so —
    unlike owner_org — this never guesses or interrupts; an unmatched license is left as-is.
    """
    configured_norm = _normalize_license(configured)
    if not configured_norm or not licenses:
        return None
    for lic in licenses:
        if str(lic.get("id") or "").strip().casefold() == str(configured).strip().casefold():
            return str(lic.get("id"))
    for lic in licenses:
        candidates = {_normalize_license(lic.get("id")), _normalize_license(lic.get("title"))} - {""}
        if configured_norm in candidates:
            return str(lic.get("id"))
    return None


def fetch_licenses(settings: Settings) -> list[dict[str, Any]]:
    """Best-effort fetch of the portal's enabled licenses (``license_list``)."""
    result = _read_client(settings).action_get("license_list")
    return result if isinstance(result, list) else []
