from __future__ import annotations

import json
import sys
from pathlib import Path

BASIC_AGENT_ROOT = Path(__file__).resolve().parents[1] / "basic-ckan-agent"
sys.path.insert(0, str(BASIC_AGENT_ROOT))

from basic_ckan_agent.files.catalog import build_file_tool_catalog  # noqa: E402
from basic_ckan_agent.files.safety import extract_file_paths  # noqa: E402
from basic_ckan_agent.files.tools import build_file_tools  # noqa: E402


def test_file_tool_catalog_documents_core_tools() -> None:
    names = {item["tool"] for item in build_file_tool_catalog()}

    assert "file_stat" in names
    assert "file_profile_csv" in names
    assert "file_profile_geojson" in names


def test_extract_file_paths_handles_relative_absolute_and_quoted_paths() -> None:
    paths = extract_file_paths("Plan metadata from sample.csv and '/tmp/flood report.pdf'.")

    assert "sample.csv" in paths
    assert "/tmp/flood report.pdf" in paths


def test_file_tools_are_limited_to_user_supplied_paths(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.txt"
    blocked = tmp_path / "blocked.txt"
    allowed.write_text("allowed content", encoding="utf-8")
    blocked.write_text("blocked content", encoding="utf-8")

    tools = _tools_by_name(build_file_tools(allowed_paths=[str(allowed)]))
    payload = json.loads(tools["file_read_text"].invoke({"path": str(blocked)}))

    assert payload["success"] is False
    assert payload["error"]["code"] == "path_not_allowed"


def test_file_read_text_returns_bounded_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Dataset notes about flood depth and survey sites.", encoding="utf-8")

    tools = _tools_by_name(build_file_tools(allowed_paths=[str(path)]))
    payload = json.loads(tools["file_read_text"].invoke({"path": str(path), "max_chars": 20}))

    assert payload["success"] is True
    assert payload["result"]["text"] == "Dataset notes about "
    assert payload["result"]["truncated"] is True


def test_file_profile_csv_returns_headers_and_sample_rows(tmp_path: Path) -> None:
    path = tmp_path / "stations.csv"
    path.write_text("station,depth\nA,1.2\nB,3.4\n", encoding="utf-8")

    tools = _tools_by_name(build_file_tools(allowed_paths=[str(path)]))
    payload = json.loads(tools["file_profile_csv"].invoke({"path": str(path), "max_rows": 2}))

    assert payload["success"] is True
    assert payload["result"]["headers"] == ["station", "depth"]
    assert payload["result"]["sample_rows"][0] == {"station": "A", "depth": "1.2"}


def test_file_profile_geojson_returns_bbox_and_spatial_polygon(tmp_path: Path) -> None:
    path = tmp_path / "sites.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"site": "A"},
                        "geometry": {"type": "Point", "coordinates": [-100.0, 30.0]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"site": "B"},
                        "geometry": {"type": "Point", "coordinates": [-99.0, 31.0]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    tools = _tools_by_name(build_file_tools(allowed_paths=[str(path)]))
    payload = json.loads(tools["file_profile_geojson"].invoke({"path": str(path)}))

    assert payload["success"] is True
    assert payload["result"]["bbox"] == [-100.0, 30.0, -99.0, 31.0]
    assert payload["result"]["spatial_geojson"]["type"] == "Polygon"
    assert payload["result"]["property_keys"] == ["site"]


def _tools_by_name(tools):
    return {tool.name: tool for tool in tools}
