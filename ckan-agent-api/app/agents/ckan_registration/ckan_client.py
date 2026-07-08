from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class CkanClient:
    def __init__(self, *, base_url: str, authorization_header: str | None = None, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.authorization_header = authorization_header
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": self.authorization_header} if self.authorization_header else {}

    def action_get(self, action: str, params: dict[str, Any] | None = None) -> Any:
        response = requests.get(
            f"{self.base_url}/api/3/action/{action}",
            params=params or {},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"CKAN {action} failed: {payload.get('error')}")
        return payload["result"]

    def action_post(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        files: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"headers": self.headers, "timeout": self.timeout}
        if files:
            kwargs["data"] = payload
            kwargs["files"] = files
        else:
            kwargs["json"] = payload
        response = requests.post(f"{self.base_url}/api/3/action/{action}", **kwargs)
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise RuntimeError(f"CKAN {action} failed: {body.get('error')}")
        return body["result"]

    def package_show(self, dataset_name: str) -> Any | None:
        try:
            return self.action_get("package_show", {"id": dataset_name})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    def package_search(self, query: str, *, rows: int = 10) -> list[dict[str, Any]]:
        result = self.action_get("package_search", {"q": query, "rows": rows})
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            return result["results"]
        return []

    def organization_show(self, organization: str) -> Any | None:
        try:
            return self.action_get("organization_show", {"id": organization})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    def organization_list(self) -> list[dict[str, Any]]:
        result = self.action_get("organization_list", {"all_fields": True})
        return result if isinstance(result, list) else []

    def organization_list_for_user(self, user_id: str = "current") -> list[dict[str, Any]] | None:
        """Return organizations the authenticated user belongs to.

        Returns None when the request fails (network error, CKAN unavailable, or
        the action is not permitted), allowing callers to degrade gracefully.
        Returns an empty list when the call succeeds but the user has no memberships.
        HTTP 401 is treated as empty (invalid/expired JWT → no access).
        """
        try:
            result = self.action_get("organization_list_for_user", {"id": user_id, "permission": "read"})
            return result if isinstance(result, list) else []
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                return []  # invalid or expired token — definitely no access
            return None  # 403 (action restricted), 404, or other HTTP error — uncertain
        except Exception:
            return None  # network error or CKAN unavailable — uncertain

    def resolve_organization_id(self, organization: str) -> dict[str, str]:
        query = str(organization or "").strip()
        if not query:
            return {"id": "", "name": "", "title": "", "matched_by": "empty"}

        shown = self.organization_show(query)
        if isinstance(shown, dict):
            return {
                "id": str(shown.get("id") or query),
                "name": str(shown.get("name") or ""),
                "title": str(shown.get("title") or shown.get("display_name") or ""),
                "matched_by": "organization_show",
            }

        normalized = query.casefold()
        for org in self.organization_list():
            candidates = [
                str(org.get("id") or ""),
                str(org.get("name") or ""),
                str(org.get("title") or ""),
                str(org.get("display_name") or ""),
            ]
            if normalized in {candidate.casefold() for candidate in candidates if candidate}:
                return {
                    "id": str(org.get("id") or ""),
                    "name": str(org.get("name") or ""),
                    "title": str(org.get("title") or org.get("display_name") or ""),
                    "matched_by": "organization_list",
                }

        return {"id": query, "name": query, "title": "", "matched_by": "unresolved"}

    def user_show_current(self) -> dict[str, Any] | None:
        """Return the authenticated user's CKAN profile, or None if unavailable.

        Passes ``id=current`` which is supported by CKAN 2.9+ when the request
        carries a valid Authorization header. Degrades gracefully on older portals
        or unauthenticated requests by returning None.
        """
        try:
            return self.action_get("user_show", {"id": "current"})
        except Exception:  # noqa: BLE001
            return None

    def resource_upload(self, payload: dict[str, Any], file_path: Path, *, update: bool = False) -> Any:
        action = "resource_update" if update else "resource_create"
        with file_path.open("rb") as handle:
            return self.action_post(action, payload, files={"upload": handle})
