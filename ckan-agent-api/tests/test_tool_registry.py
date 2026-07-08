from __future__ import annotations

from pathlib import Path

import pytest

from app.settings import PROJECT_ROOT
from app.tools import ToolError, ToolRegistry

SEED_TOOLS_DIR = PROJECT_ROOT / "app" / "tools" / "catalog"


def _registry() -> ToolRegistry:
    return ToolRegistry(SEED_TOOLS_DIR)


def test_seed_catalog_loads_ckan_and_file_tools():
    names = {s.name for s in _registry().load_all()}
    assert {"ckan_package_show", "ckan_package_search", "ckan_dry_run_diff"} <= names
    assert {"file_profile_csv", "file_profile_geojson", "file_extract_pdf_text"} <= names


def test_openai_tool_schema_shape():
    schema = _registry().get("ckan_package_search").to_openai_tool()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "ckan_package_search"
    assert fn["parameters"]["required"] == ["query"]
    assert "rows" in fn["parameters"]["properties"]


def test_to_openai_tools_filter_by_name():
    tools = _registry().to_openai_tools(names=["ckan_package_search"])
    assert [t["function"]["name"] for t in tools] == ["ckan_package_search"]


def test_invoke_file_tool_success(tmp_path: Path):
    f = tmp_path / "d.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    out = _registry().invoke("file_profile_csv", {"path": str(f)})
    assert out["success"] is True
    assert out["tool"] == "file_profile_csv"
    assert out["result"]["headers"] == ["a", "b"]


def test_invoke_missing_required_arg():
    out = _registry().invoke("file_profile_csv", {})
    assert out["success"] is False
    assert out["error"]["code"] == "invalid_args"


def test_invoke_unknown_tool():
    out = _registry().invoke("nope", {})
    assert out["success"] is False
    assert out["error"]["code"] == "unknown_tool"


def test_invoke_handler_error_on_sensitive_path(tmp_path: Path):
    f = tmp_path / "secret.pem"
    f.write_text("-----BEGIN KEY-----", encoding="utf-8")
    out = _registry().invoke("file_read_text", {"path": str(f)})
    assert out["success"] is False
    assert out["error"]["code"] == "handler_error"


def test_invoke_ckan_tool_with_mocked_client(monkeypatch):
    class FakeClient:
        def package_search(self, query, *, rows=10):
            return [{"name": "ds-1", "title": query}]

    monkeypatch.setattr("app.tools.handlers.ckan._read_client", lambda: FakeClient())
    out = _registry().invoke("ckan_package_search", {"query": "groundwater"})
    assert out["success"] is True
    assert out["result"][0]["name"] == "ds-1"


def test_write_tool_is_refused_at_load(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: ckan_delete\nsummary: deletes\nhandler: app.tools.handlers.ckan:package_show\nread_only: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ToolError, match="read_only=false"):
        ToolRegistry(tmp_path).load_all()
