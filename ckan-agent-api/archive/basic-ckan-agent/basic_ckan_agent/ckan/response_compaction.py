from __future__ import annotations

from typing import Any


def compact_package(pkg: dict[str, Any], *, mode: str = "detail") -> dict[str, Any]:
    resources = pkg.get("resources") or []
    tags = pkg.get("tags") or []

    if mode == "search":
        return {
            "id": pkg.get("id"),
            "name": pkg.get("name"),
            "title": pkg.get("title"),
            "notes_preview": (pkg.get("notes") or "")[:300],
            "organization": (
                pkg.get("organization", {}).get("title")
                if isinstance(pkg.get("organization"), dict)
                else None
            ),
            "tags": [
                tag.get("name")
                for tag in tags[:10]
                if isinstance(tag, dict) and tag.get("name")
            ],
            "resource_count": len(resources),
            "resource_formats": sorted(
                {
                    r.get("format")
                    for r in resources
                    if isinstance(r, dict) and r.get("format")
                }
            ),
        }

    return {
        "id": pkg.get("id"),
        "name": pkg.get("name"),
        "title": pkg.get("title"),
        "notes": pkg.get("notes"),
        "private": pkg.get("private"),
        "owner_org": pkg.get("owner_org"),
        "organization": (
            {
                "name": pkg.get("organization", {}).get("name"),
                "title": pkg.get("organization", {}).get("title"),
            }
            if isinstance(pkg.get("organization"), dict)
            else None
        ),
        "license_id": pkg.get("license_id"),
        "license_title": pkg.get("license_title"),
        "tags": [
            tag.get("name")
            for tag in tags
            if isinstance(tag, dict) and tag.get("name")
        ],
        "resource_count": len(resources),
        "resources": [
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "format": r.get("format"),
                "url": r.get("url"),
            }
            for r in resources[:5]
            if isinstance(r, dict)
        ],
    }


def compact_ckan_response(action_name: str, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"success": False, "raw_type": type(data).__name__}

    compact: dict[str, Any] = {"success": data.get("success")}

    if data.get("error"):
        compact["error"] = data.get("error")
        return compact

    result = data.get("result")

    if action_name == "package_search" and isinstance(result, dict):
        results = result.get("results") or []
        compact["result"] = {
            "count": result.get("count"),
            "results_returned": len(results),
            "results": [
                compact_package(pkg, mode="search")
                for pkg in results[:10]
                if isinstance(pkg, dict)
            ],
        }
        return compact

    if action_name in {"package_show", "package_create", "package_update", "package_patch"}:
        compact["result"] = compact_package(result, mode="detail") if isinstance(result, dict) else result
        return compact

    if action_name == "current_package_list_with_resources" and isinstance(result, list):
        compact["result"] = [
            compact_package(pkg)
            for pkg in result[:10]
            if isinstance(pkg, dict)
        ]
        compact["results_returned"] = len(compact["result"])
        return compact

    if action_name == "package_list" and isinstance(result, list):
        compact["result_count"] = len(result)
        compact["result"] = result[:50]
        return compact

    if action_name == "organization_list" and isinstance(result, list):
        compact["result_count"] = len(result)
        compact["result"] = [
            {
                "id": org.get("id"),
                "name": org.get("name"),
                "title": org.get("title"),
                "package_count": org.get("package_count"),
            }
            if isinstance(org, dict)
            else org
            for org in result[:50]
        ]
        return compact

    if action_name == "resource_show" and isinstance(result, dict):
        compact["result"] = {
            "id": result.get("id"),
            "package_id": result.get("package_id"),
            "name": result.get("name"),
            "description": result.get("description"),
            "format": result.get("format"),
            "mimetype": result.get("mimetype"),
            "url": result.get("url"),
            "size": result.get("size"),
            "created": result.get("created"),
            "last_modified": result.get("last_modified"),
        }
        return compact

    if action_name == "status_show" and isinstance(result, dict):
        compact["result"] = {
            "site_title": result.get("site_title"),
            "site_url": result.get("site_url"),
            "ckan_version": result.get("ckan_version"),
            "extensions": result.get("extensions"),
        }
        return compact

    compact["result"] = result
    return compact

