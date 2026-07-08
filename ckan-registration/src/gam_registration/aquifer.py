"""Aquifer-boundary bbox fallback for GAM spatial derivation.

Fetches major/minor Texas aquifer polygons from TWDB ArcGIS FeatureServer
endpoints discovered via the Tapis STAC API. Caches statewide GeoJSON in
memory so the download happens at most once per Python process.

Usage::

    from aquifer import get_aquifer_bbox

    result = get_aquifer_bbox("Blossom", kind="minor")
    if result is not None:
        bbox, geojson_polygon = result
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_BASE_URL = "https://stacapi.pods.portals.tapis.io/api/v1"
STAC_COLLECTION = "subside-context"
STAC_ITEM_MAJOR = "major-aquifers"
STAC_ITEM_MINOR = "minor-aquifers"

# Primary aquifer-name field confirmed via FeatureServer/0?f=json inspection.
# Alternates: AQ_NAME_UL (string), AQUIFER (int code).
AQUIFER_NAME_FIELD = "AQU_NAME"

# In-memory cache: keyed by "major" or "minor"
_geojson_cache: dict[str, dict[str, Any]] = {}
_stac_href_cache: dict[str, str] = {}  # keyed by "major" | "minor"


# ---------------------------------------------------------------------------
# STAC helpers
# ---------------------------------------------------------------------------


def _fetch_stac_asset_href(kind: str, *, timeout: int = 30) -> str:
    """Return the ArcGIS FeatureServer href from the STAC item for *kind*.

    Args:
        kind: ``"major"`` or ``"minor"``.
        timeout: HTTP request timeout in seconds.

    Returns:
        The ``asset["service"]["href"]`` string from the STAC item.

    Raises:
        KeyError: if the expected asset key is absent.
        requests.HTTPError: on non-2xx responses.
    """
    if kind in _stac_href_cache:
        return _stac_href_cache[kind]

    item_id = STAC_ITEM_MAJOR if kind == "major" else STAC_ITEM_MINOR
    url = f"{STAC_BASE_URL}/collections/{STAC_COLLECTION}/items/{item_id}"
    logger.debug("Fetching STAC item %s from %s", item_id, url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    item = resp.json()

    assets = item.get("assets", {})
    service_asset = assets.get("service", {})
    href = service_asset.get("href")
    if not href:
        raise KeyError(
            f"STAC item '{item_id}' has no assets.service.href; "
            f"available assets: {list(assets.keys())}"
        )

    _stac_href_cache[kind] = href
    logger.info("STAC asset href for %s aquifers: %s", kind, href)
    return href


# ---------------------------------------------------------------------------
# ArcGIS FeatureServer fetch (cached per kind)
# ---------------------------------------------------------------------------


def _fetch_aquifer_geojson(kind: str, *, timeout: int = 60) -> dict[str, Any]:
    """Fetch the statewide aquifer GeoJSON for *kind*, with in-memory cache.

    Args:
        kind: ``"major"`` or ``"minor"``.
        timeout: HTTP request timeout in seconds.

    Returns:
        A GeoJSON FeatureCollection dict.
    """
    if kind in _geojson_cache:
        return _geojson_cache[kind]

    href = _fetch_stac_asset_href(kind, timeout=timeout)
    logger.info("Downloading %s aquifer GeoJSON from %s", kind, href)
    resp = requests.get(href, timeout=timeout)
    resp.raise_for_status()
    fc = resp.json()

    _geojson_cache[kind] = fc
    feature_count = len(fc.get("features", []))
    logger.info(
        "Cached %s aquifer GeoJSON: %d features", kind, feature_count
    )
    return fc


# ---------------------------------------------------------------------------
# Coordinate extraction helpers
# ---------------------------------------------------------------------------


def _extract_ring_coords(geometry: dict[str, Any]) -> list[list[float]]:
    """Flatten all coordinate pairs from a Polygon or MultiPolygon geometry.

    Args:
        geometry: A GeoJSON geometry dict with ``type`` and ``coordinates``.

    Returns:
        A flat list of ``[lon, lat]`` coordinate pairs.
    """
    geo_type = geometry.get("type", "")
    coords_field = geometry.get("coordinates", [])
    pairs: list[list[float]] = []

    if geo_type == "Polygon":
        # coordinates = [ ring, ring, ... ]  each ring = [[lon, lat], ...]
        for ring in coords_field:
            pairs.extend(ring)
    elif geo_type == "MultiPolygon":
        # coordinates = [ polygon, polygon, ... ]
        # polygon = [ ring, ring, ... ]
        for polygon in coords_field:
            for ring in polygon:
                pairs.extend(ring)
    else:
        logger.warning(
            "Unsupported geometry type '%s'; cannot extract coordinates.", geo_type
        )

    return pairs


def _bbox_from_coords(
    pairs: list[list[float]],
) -> dict[str, float] | None:
    """Compute a WGS84 bounding box from a list of ``[lon, lat]`` pairs.

    Args:
        pairs: List of ``[lon, lat]`` pairs.

    Returns:
        Dict with ``min_lon``, ``min_lat``, ``max_lon``, ``max_lat``,
        or ``None`` if *pairs* is empty.
    """
    if not pairs:
        return None
    lons = [p[0] for p in pairs]
    lats = [p[1] for p in pairs]
    return {
        "min_lon": min(lons),
        "min_lat": min(lats),
        "max_lon": max(lons),
        "max_lat": max(lats),
    }


def _bbox_to_geojson_polygon(bbox: dict[str, float]) -> dict[str, Any]:
    """Convert a ``{min_lon, min_lat, max_lon, max_lat}`` bbox to a GeoJSON Polygon.

    The polygon ring is closed (first point == last point) and uses the
    standard counter-clockwise winding order for the exterior ring.

    Args:
        bbox: Dict with ``min_lon``, ``min_lat``, ``max_lon``, ``max_lat``.

    Returns:
        A GeoJSON Polygon dict.
    """
    min_lon = bbox["min_lon"]
    min_lat = bbox["min_lat"]
    max_lon = bbox["max_lon"]
    max_lat = bbox["max_lat"]
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ]
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_aquifer_bbox(
    aquifer_name: str | list[str],
    *,
    kind: str = "minor",
    timeout: int = 60,
) -> tuple[dict[str, float], dict[str, Any]] | None:
    """Look up the WGS84 bounding box for one or more named Texas aquifers.

    Fetches the statewide aquifer GeoJSON from the TWDB ArcGIS FeatureServer
    (URL read from the Tapis STAC item — not hardcoded), caches it in memory,
    then filters features case-insensitively on ``AQU_NAME``.

    For multi-aquifer lookups (``aquifer_name`` is a list), the returned bbox
    is the union of all matched feature extents.

    Args:
        aquifer_name: Aquifer name string, or list of names for multi-aquifer
            GAMs. Each name is matched case-insensitively against the
            ``AQU_NAME`` field.
        kind: ``"major"`` or ``"minor"`` — selects which STAC item / layer to
            query. If a model spans both kinds, call this function twice and
            union the results manually.
        timeout: HTTP timeout in seconds applied to STAC and ArcGIS requests.

    Returns:
        A ``(bbox_dict, geojson_polygon)`` tuple on success, where
        ``bbox_dict`` has ``min_lon``, ``min_lat``, ``max_lon``, ``max_lat``
        and ``geojson_polygon`` is a GeoJSON Polygon dict.
        Returns ``None`` on any network error or if no features match.
    """
    names: list[str]
    if isinstance(aquifer_name, str):
        names = [aquifer_name]
    else:
        names = list(aquifer_name)

    if not names:
        logger.warning("get_aquifer_bbox called with empty aquifer_name; returning None.")
        return None

    try:
        fc = _fetch_aquifer_geojson(kind, timeout=timeout)
    except Exception as exc:
        logger.warning(
            "Failed to fetch %s aquifer GeoJSON: %s", kind, exc
        )
        return None

    features = fc.get("features", [])
    names_upper = {n.upper() for n in names}

    all_pairs: list[list[float]] = []
    matched_names: list[str] = []

    for feature in features:
        props = feature.get("properties") or {}
        # Primary field; fall back to AQ_NAME_UL if AQU_NAME absent
        aqu_name_val = props.get(AQUIFER_NAME_FIELD) or props.get("AQ_NAME_UL", "")
        if str(aqu_name_val).upper() in names_upper:
            geometry = feature.get("geometry")
            if geometry:
                pairs = _extract_ring_coords(geometry)
                all_pairs.extend(pairs)
                matched_names.append(str(aqu_name_val))

    if not all_pairs:
        logger.warning(
            "No %s aquifer features matched names %s; returning None.",
            kind,
            names,
        )
        return None

    logger.info(
        "Matched %s aquifer features for %s: %s",
        kind,
        names,
        list(set(matched_names)),
    )

    bbox = _bbox_from_coords(all_pairs)
    if bbox is None:
        logger.warning("Matched features yielded no coordinate pairs; returning None.")
        return None

    geojson_polygon = _bbox_to_geojson_polygon(bbox)
    return bbox, geojson_polygon


def get_aquifer_bbox_for_model(
    aquifer_name: str | list[str],
    *,
    kind: str,
    timeout: int = 60,
) -> tuple[dict[str, float], dict[str, Any]] | None:
    """Convenience wrapper used by ``discovery.py`` to look up a model's aquifer bbox.

    Identical to :func:`get_aquifer_bbox`; provided so callers import a
    single named entry point rather than passing ``kind=`` as a kwarg each time.

    Args:
        aquifer_name: Aquifer name or list of names.
        kind: ``"major"`` or ``"minor"``.
        timeout: HTTP timeout in seconds.

    Returns:
        ``(bbox_dict, geojson_polygon)`` or ``None``.
    """
    return get_aquifer_bbox(aquifer_name, kind=kind, timeout=timeout)


def clear_cache() -> None:
    """Clear the in-memory STAC and GeoJSON caches (primarily for testing)."""
    _geojson_cache.clear()
    _stac_href_cache.clear()
