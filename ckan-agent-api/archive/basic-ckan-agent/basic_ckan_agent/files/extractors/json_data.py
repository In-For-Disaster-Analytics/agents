from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GEOJSON_TYPES = {
    "FeatureCollection",
    "Feature",
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
}


def profile_json(path: Path, *, max_sample_chars: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    sample = _compact_sample(data)
    sample_text = json.dumps(sample, ensure_ascii=False, default=str)
    if len(sample_text) > max_sample_chars:
        sample_text = sample_text[:max_sample_chars]

    return {
        "json_type": type(data).__name__,
        "summary": _summarize_value(data),
        "sample_json": sample_text,
        "geojson_hint": _geojson_hint(data),
    }


def profile_geojson(path: Path, *, max_sample_chars: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    geojson_type = data.get("type") if isinstance(data, dict) else None
    if not isinstance(data, dict) or not isinstance(geojson_type, str):
        return {
            "is_geojson": False,
            "message": "JSON root is not a GeoJSON-like object with a type field.",
            "json_profile": profile_json(path, max_sample_chars=max_sample_chars),
        }

    features = _features(data)
    geometries = _geometries(data, features)
    bbox = _bbox_for_geometries(geometries)
    property_counter: Counter[str] = Counter()
    for feature in features:
        props = feature.get("properties") if isinstance(feature, dict) else None
        if isinstance(props, dict):
            property_counter.update(str(key) for key in props.keys())

    sample_features = _compact_sample(features[:5])
    sample_text = json.dumps(sample_features, ensure_ascii=False, default=str)
    if len(sample_text) > max_sample_chars:
        sample_text = sample_text[:max_sample_chars]

    return {
        "is_geojson": geojson_type in GEOJSON_TYPES,
        "geojson_type": geojson_type,
        "feature_count": len(features) if features else (1 if geojson_type == "Feature" else None),
        "geometry_types": sorted(
            {geometry.get("type") for geometry in geometries if isinstance(geometry, dict) and geometry.get("type")}
        ),
        "bbox": bbox,
        "spatial_geojson": _bbox_polygon_geojson(bbox) if bbox else None,
        "property_keys": [key for key, _ in property_counter.most_common(100)],
        "sample_features_json": sample_text,
        "crs": data.get("crs"),
    }


def _summarize_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": list(value.keys())[:100],
            "key_count": len(value),
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "item_types": sorted({type(item).__name__ for item in value[:100]}),
        }
    return {"type": type(value).__name__}


def _compact_sample(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return _scalar_preview(value)
    if isinstance(value, dict):
        return {str(key): _compact_sample(item, depth=depth + 1) for key, item in list(value.items())[:25]}
    if isinstance(value, list):
        return [_compact_sample(item, depth=depth + 1) for item in value[:10]]
    return _scalar_preview(value)


def _scalar_preview(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "..."
    return value


def _geojson_hint(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"looks_like_geojson": False}
    return {
        "looks_like_geojson": data.get("type") in GEOJSON_TYPES,
        "type": data.get("type"),
    }


def _features(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("type") == "FeatureCollection" and isinstance(data.get("features"), list):
        return [feature for feature in data["features"] if isinstance(feature, dict)]
    if data.get("type") == "Feature":
        return [data]
    return []


def _geometries(data: dict[str, Any], features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if features:
        return [
            geometry
            for feature in features
            if isinstance(feature.get("geometry"), dict)
            for geometry in [feature["geometry"]]
        ]
    if isinstance(data.get("coordinates"), list) or data.get("type") == "GeometryCollection":
        return [data]
    return []


def _bbox_for_geometries(geometries: list[dict[str, Any]]) -> list[float] | None:
    positions: list[tuple[float, float]] = []
    for geometry in geometries:
        positions.extend(_positions_for_geometry(geometry))
    if not positions:
        return None
    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]
    return [min(xs), min(ys), max(xs), max(ys)]


def _positions_for_geometry(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    if geometry.get("type") == "GeometryCollection" and isinstance(geometry.get("geometries"), list):
        positions: list[tuple[float, float]] = []
        for child in geometry["geometries"]:
            if isinstance(child, dict):
                positions.extend(_positions_for_geometry(child))
        return positions
    return list(_positions(geometry.get("coordinates")))


def _positions(value: Any) -> Iterable[tuple[float, float]]:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], int | float)
        and isinstance(value[1], int | float)
    ):
        yield (float(value[0]), float(value[1]))
        return
    if isinstance(value, list):
        for child in value:
            yield from _positions(child)


def _bbox_polygon_geojson(bbox: list[float]) -> dict[str, Any]:
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }
