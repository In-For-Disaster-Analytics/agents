from __future__ import annotations

import json
from pathlib import Path

import app.tools.handlers.gdal as g
from app.settings import PROJECT_ROOT
from app.tools import ToolRegistry


def test_gdal_tools_registered():
    names = {s.name for s in ToolRegistry(PROJECT_ROOT / "app" / "tools" / "catalog").load_all()}
    assert {"gdal_info", "ogr_info"} <= names


def test_gdal_info_missing_dependency(monkeypatch):
    monkeypatch.setattr(g.shutil, "which", lambda _name: None)
    out = g.gdal_info({"path": "/nope.tif"})
    assert out["dependency_missing"] == "gdal"


def test_gdal_info_parses_output(monkeypatch, tmp_path: Path):
    raster = tmp_path / "r.tif"
    raster.write_text("x", encoding="utf-8")
    monkeypatch.setattr(g.shutil, "which", lambda _name: "/usr/bin/gdalinfo")

    class _Proc:
        returncode = 0
        stdout = json.dumps(
            {
                "driverShortName": "GTiff",
                "size": [10, 20],
                "coordinateSystem": {"wkt": "GEOGCS[EPSG:4326]"},
                "cornerCoordinates": {"upperLeft": [0, 0]},
                "bands": [{}, {}, {}],
            }
        )
        stderr = ""

    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: _Proc())
    out = g.gdal_info({"path": str(raster)})
    assert out["driver"] == "GTiff"
    assert out["size"] == [10, 20]
    assert out["band_count"] == 3
    assert "EPSG:4326" in out["crs_wkt"]
