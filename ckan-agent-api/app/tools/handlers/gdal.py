"""GDAL CLI tool handlers (spec Increment 8b).

Run ``gdalinfo`` / ``ogrinfo`` via subprocess (args list, never a shell string) when GDAL is on
PATH; degrade gracefully with ``dependency_missing`` otherwise. The path is safety-validated
(size + sensitive-path guard) before invoking GDAL, and output is bounded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from app.files.safety import validate_readable_file

_TIMEOUT = 60
_MAX_OUTPUT = 4000


def _safe_path(args: dict[str, Any]) -> str:
    return str(validate_readable_file(str(args["path"])).path)


def gdal_info(args: dict[str, Any]) -> dict[str, Any]:
    """Raster metadata (driver, size, CRS, corner coords, band count) via `gdalinfo -json`."""
    if not shutil.which("gdalinfo"):
        return {"dependency_missing": "gdal", "message": "gdalinfo is not on PATH."}
    path = _safe_path(args)
    proc = subprocess.run(["gdalinfo", "-json", path], capture_output=True, text=True, timeout=_TIMEOUT)
    if proc.returncode != 0:
        return {"error": (proc.stderr or "gdalinfo failed").strip()[:_MAX_OUTPUT]}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"raw": proc.stdout[:_MAX_OUTPUT]}
    cs = data.get("coordinateSystem") or {}
    return {
        "driver": data.get("driverShortName"),
        "size": data.get("size"),
        "crs_wkt": str(cs.get("wkt") or "")[:1500],
        "corner_coordinates": data.get("cornerCoordinates"),
        "band_count": len(data.get("bands") or []),
    }


def ogr_info(args: dict[str, Any]) -> dict[str, Any]:
    """Vector summary (layers, geometry, CRS, extent) via `ogrinfo -so -al`."""
    if not shutil.which("ogrinfo"):
        return {"dependency_missing": "gdal", "message": "ogrinfo is not on PATH."}
    path = _safe_path(args)
    proc = subprocess.run(["ogrinfo", "-so", "-al", path], capture_output=True, text=True, timeout=_TIMEOUT)
    if proc.returncode != 0:
        return {"error": (proc.stderr or "ogrinfo failed").strip()[:_MAX_OUTPUT]}
    return {"summary": proc.stdout[:_MAX_OUTPUT]}
