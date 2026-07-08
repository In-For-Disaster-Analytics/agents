from __future__ import annotations

import json
from pathlib import Path

from app.files import analyze_path, analyze_request_files, build_file_inventory, gather_file_evidence


def test_csv_routed_to_tabular(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("lon,lat,value\n-97.7,30.3,1\n-97.8,30.4,2\n", encoding="utf-8")
    report = analyze_path(f)
    assert report["format"] == "CSV"
    assert report["tabular"]["headers"] == ["lon", "lat", "value"]
    assert report["tabular"]["column_count"] == 3


def test_geojson_routed_with_bbox(tmp_path: Path):
    f = tmp_path / "pts.geojson"
    f.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"name": "a"}, "geometry": {"type": "Point", "coordinates": [-97.7, 30.3]}}
                ],
            }
        ),
        encoding="utf-8",
    )
    report = analyze_path(f)
    assert report["geojson"]["is_geojson"] is True
    assert report["geojson"]["bbox"] == [-97.7, 30.3, -97.7, 30.3]


def test_text_routed_to_text_sample(tmp_path: Path):
    f = tmp_path / "readme.md"
    f.write_text("# Title\nSome description.\n", encoding="utf-8")
    report = analyze_path(f)
    assert "# Title" in report["text"]["text"]


def test_sensitive_suffix_refused(tmp_path: Path):
    f = tmp_path / "server.pem"
    f.write_text("-----BEGIN PRIVATE KEY-----", encoding="utf-8")
    report = analyze_path(f)
    assert report["error"]["code"] == "sensitive_path"


def test_oversized_file_refused(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CKAN_AGENT_MAX_FILE_BYTES", "10")
    f = tmp_path / "big.txt"
    f.write_text("x" * 100, encoding="utf-8")
    report = analyze_path(f)
    assert report["error"]["code"] == "file_too_large"


def test_ipynb_saved_as_txt_is_parsed_as_notebook(tmp_path: Path):
    nb = {
        "nbformat": 4,
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": [
            {"cell_type": "markdown", "source": ["# My Analysis\n", "A description."]},
            {"cell_type": "code", "source": ["import folium\n", "import pandas as pd\n"]},
        ],
    }
    f = tmp_path / "notebook.txt"  # .txt extension, ipynb content
    f.write_text(json.dumps(nb), encoding="utf-8")
    report = analyze_path(f)
    assert "notebook" in report["json"]
    info = report["json"]["notebook"]
    assert info["code_cell_count"] == 1
    assert "folium" in info["python_imports"]
    assert "My Analysis" in info["markdown_headings"]


def test_notebook_profile_surfaces_code_so_bbox_is_readable(tmp_path: Path):
    """The code cells (where a bbox / date / CRS live) reach the author via code_preview,
    not just imports/headings."""
    nb = {
        "nbformat": 4,
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": [
            {"cell_type": "markdown", "source": ["# OPERA DISP-S1\n"]},
            {"cell_type": "code", "source": ["bbox = [-97.5, 32.6, -96.9, 33.1]\n", "start_year = 2016\n"]},
        ],
    }
    f = tmp_path / "analysis.ipynb"
    f.write_text(json.dumps(nb), encoding="utf-8")
    info = analyze_path(f)["json"]["notebook"]
    assert "code_preview" in info
    assert "bbox = [-97.5, 32.6, -96.9, 33.1]" in info["code_preview"]
    assert "2016" in info["code_preview"]
    assert info["code_truncated"] is False


def test_pdf_extractor_rejects_non_pdf_instead_of_crashing(tmp_path: Path):
    from app.files.extractors.pdf import extract_pdf_text, looks_like_pdf

    f = tmp_path / "notebook.ipynb"
    f.write_text('{\n "cells": []\n}', encoding="utf-8")
    assert looks_like_pdf(f) is False
    report = extract_pdf_text(f, page_start=0, max_pages=12, max_chars=12000)
    assert report["error"] == "not_a_pdf"
    assert report["text"] == ""


def test_read_text_tool_returns_notebook_source(tmp_path: Path):
    from app.tools.handlers.files import read_text

    nb = {
        "nbformat": 4,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n"]},
            {"cell_type": "code", "source": ["aoi = [-100, 30, -99, 31]\n"]},
        ],
    }
    f = tmp_path / "nb.ipynb"
    f.write_text(json.dumps(nb), encoding="utf-8")
    out = read_text({"path": str(f)})
    assert out["notebook"] is True
    assert "aoi = [-100, 30, -99, 31]" in out["text"]


def test_inline_attached_notebook_is_analyzed():
    nb = {
        "nbformat": 4,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "source": ["# Flood Map\n"]},
            {"cell_type": "code", "source": ["import folium\n"]},
        ],
    }
    reports, _ = analyze_request_files({"inline_files": [{"name": "demo.ipynb", "content": json.dumps(nb)}]})
    assert len(reports) == 1
    assert reports[0]["inline"] is True
    assert reports[0]["name"] == "demo.ipynb"
    assert "folium" in reports[0]["json"]["notebook"]["python_imports"]
    assert "Flood Map" in reports[0]["json"]["notebook"]["markdown_headings"]


def test_gather_file_evidence_heads_always_full_only_when_few(tmp_path: Path):
    d = tmp_path / "u"
    d.mkdir()
    for i in range(5):
        (d / f"f{i}.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    many = gather_file_evidence({"upload_dir": str(d)}, deep_threshold=3)
    assert len(many["file_heads"]) == 5  # heads for all
    assert many["file_reports"] == []  # 5 > 3 -> no up-front full analysis (author tool-calls)

    few = gather_file_evidence({"upload_dir": str(d)}, deep_threshold=10)
    assert len(few["file_reports"]) == 5  # <= threshold -> fully analyzed up front


def test_analyze_request_files_expands_upload_dir_and_inventory(tmp_path: Path):
    d = tmp_path / "upload"
    d.mkdir()
    (d / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (d / "b.json").write_text("{}", encoding="utf-8")
    reports, warnings = analyze_request_files({"upload_dir": str(d)})
    assert len(reports) == 2
    inventory = build_file_inventory(reports)
    assert inventory["file_count"] == 2
    assert set(inventory["extension_counts"]) == {".csv", ".json"}
