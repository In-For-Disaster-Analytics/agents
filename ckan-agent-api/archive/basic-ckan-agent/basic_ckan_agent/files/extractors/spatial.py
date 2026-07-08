from __future__ import annotations

from pathlib import Path
from typing import Any

from basic_ckan_agent.files.extractors.archive import inspect_zip


def profile_raster(path: Path) -> dict[str, Any]:
    try:
        import rasterio
    except ImportError:
        return {
            "dependency_missing": "rasterio",
            "message": "Raster profiling needs rasterio. Install it to return CRS, bounds, bands, and dtype.",
        }

    with rasterio.open(path) as dataset:
        return {
            "driver": dataset.driver,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtypes": list(dataset.dtypes),
            "crs": str(dataset.crs) if dataset.crs else None,
            "bounds": {
                "left": dataset.bounds.left,
                "bottom": dataset.bounds.bottom,
                "right": dataset.bounds.right,
                "top": dataset.bounds.top,
            },
            "nodata": dataset.nodata,
        }


def profile_shapefile_zip(path: Path) -> dict[str, Any]:
    archive_profile = inspect_zip(path, max_members=500)
    if not archive_profile.get("is_zipfile"):
        return archive_profile

    try:
        import fiona
    except ImportError:
        return {
            **archive_profile,
            "dependency_missing": "fiona",
            "message": "Shapefile ZIP structure was inspected, but schema/extent profiling needs fiona.",
        }

    uri = f"zip://{path}"
    try:
        layers = fiona.listlayers(uri)
    except Exception as exc:
        return {
            **archive_profile,
            "profile_error": str(exc),
            "message": "Fiona could not open the ZIP as a vector dataset.",
        }

    layer_profiles: list[dict[str, Any]] = []
    for layer in layers:
        try:
            with fiona.open(uri, layer=layer) as collection:
                layer_profiles.append(
                    {
                        "name": layer,
                        "driver": collection.driver,
                        "feature_count": len(collection),
                        "crs": collection.crs_wkt or collection.crs,
                        "bounds": list(collection.bounds) if collection.bounds else None,
                        "schema": collection.schema,
                    }
                )
        except Exception as exc:
            layer_profiles.append({"name": layer, "profile_error": str(exc)})

    return {**archive_profile, "layers": layer_profiles}
