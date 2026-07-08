"""Tests for the dso_ckan_mcp integration (spec 2026-06-29).

Covers schema normalization, an end-to-end MCP client round-trip over FastMCP's in-memory
transport, the write-tool dry-run hard-block, composite routing, and graceful fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.tools.executor import CompositeToolExecutor, InProcessToolExecutor, MCPToolExecutor
from app.tools.mcp_client import MCPClient, normalize_input_schema

# ── schema normalization ──────────────────────────────────────────────────


def test_normalize_resolves_refs_and_flattens_optionals():
    schema = {
        "type": "object",
        "title": "Args",
        "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}, "title": "Inner"}},
        "properties": {
            "q": {"type": "string", "title": "Q"},
            "opt": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "maybe"},
            "nested": {"$ref": "#/$defs/Inner"},
        },
        "required": ["q"],
    }
    out = normalize_input_schema(schema, hidden_args=frozenset())
    assert out["type"] == "object"
    assert "title" not in out["properties"]["q"]  # stripped
    assert out["properties"]["opt"]["type"] == "string"  # anyOf flattened
    assert out["properties"]["opt"]["nullable"] is True
    assert out["properties"]["nested"]["properties"]["x"]["type"] == "integer"  # $ref inlined
    assert out["required"] == ["q"]


def test_normalize_drops_hidden_args():
    schema = {
        "type": "object",
        "properties": {
            "metadata": {"type": "object"},
            "tapis_token": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        "required": ["metadata", "tapis_token"],
    }
    out = normalize_input_schema(schema, hidden_args=frozenset({"tapis_token", "dry_run"}))
    assert "tapis_token" not in out["properties"]
    assert "dry_run" not in out["properties"]
    assert out["required"] == ["metadata"]


# ── in-memory MCP server round-trip ─────────────────────────────────────────


@pytest.fixture()
def mcp_client():
    fastmcp = pytest.importorskip("fastmcp")

    server = fastmcp.FastMCP("test-ckan")

    @server.tool()
    def package_search(query: str, rows: int = 10) -> dict[str, Any]:
        """Search datasets."""
        return {"count": 1, "results": [{"name": "demo", "q": query, "rows": rows}]}

    @server.tool()
    def schema_create_package(
        dataset_type: str, metadata: dict[str, Any], dry_run: bool = True, tapis_token: str | None = None
    ) -> dict[str, Any]:
        """Create a package (gated write)."""
        return {"dry_run": dry_run, "token_seen": tapis_token is not None, "type": dataset_type}

    @server.prompt()
    def analyze_dataset(dataset_id: str) -> str:
        return f"Analyze {dataset_id}."

    client = MCPClient(server, timeout=10.0)
    client.connect()
    yield client
    client.close()


def test_list_and_call_read_tool(mcp_client):
    names = mcp_client.tool_names()
    assert {"package_search", "schema_create_package"} <= set(names)
    result = mcp_client.call_tool("package_search", {"query": "rain", "rows": 5})
    assert result["results"][0]["q"] == "rain"


def test_openai_schemas_scrub_hidden_and_dry_run(mcp_client):
    specs = mcp_client.to_openai_tools()
    by_name = {s["function"]["name"]: s for s in specs}
    write_params = by_name["schema_create_package"]["function"]["parameters"]["properties"]
    assert "tapis_token" not in write_params  # hidden globally
    assert "dry_run" not in write_params  # hidden for write tools (model cannot set it)
    read_params = by_name["package_search"]["function"]["parameters"]["properties"]
    assert "query" in read_params


def test_get_prompt_round_trip(mcp_client):
    text = mcp_client.get_prompt("analyze_dataset", {"dataset_id": "abc"})
    assert "Analyze abc" in text


# ── write hard-block + routing ───────────────────────────────────────────────


def test_mcp_executor_blocks_live_write(mcp_client):
    executor = MCPToolExecutor(mcp_client)
    # dry_run omitted (server default would be False at the wire) → blocked
    blocked = executor.invoke("schema_create_package", {"dataset_type": "d", "metadata": {}})
    assert blocked["success"] is False
    assert blocked["error"]["code"] == "live_write_blocked"

    # explicit live write → blocked
    blocked2 = executor.invoke(
        "schema_create_package", {"dataset_type": "d", "metadata": {}, "dry_run": False}
    )
    assert blocked2["error"]["code"] == "live_write_blocked"


def test_mcp_executor_allows_dry_run_and_strips_token(mcp_client):
    executor = MCPToolExecutor(mcp_client)
    ok = executor.invoke(
        "schema_create_package",
        {"dataset_type": "d", "metadata": {}, "dry_run": True, "tapis_token": "SECRET"},
    )
    assert ok["success"] is True
    assert ok["result"]["dry_run"] is True
    assert ok["result"]["token_seen"] is False  # tapis_token stripped before the call


def test_mcp_executor_read_tool_success(mcp_client):
    executor = MCPToolExecutor(mcp_client)
    out = executor.invoke("package_search", {"query": "x"})
    assert out["success"] is True


def test_composite_routes_by_name(mcp_client):
    class _Fake(InProcessToolExecutor):
        def __init__(self):
            pass

        def invoke(self, name, args):
            return {"success": True, "tool": name, "result": "in_process"}

    composite = CompositeToolExecutor(_Fake(), {"package_search": MCPToolExecutor(mcp_client)})
    assert composite.invoke("package_search", {"query": "x"})["result"]["results"][0]["name"] == "demo"
    assert composite.invoke("file_read_text", {"path": "/x"})["result"] == "in_process"


# ── graceful fallback ────────────────────────────────────────────────────────


def test_tool_kwargs_fallback_when_mcp_disabled():
    from app.agents.ckan_registration.persona_nodes import _mcp_executor_and_schemas
    from app.settings import Settings
    from app.tools import ToolRegistry

    settings = Settings()  # mcp_enabled defaults False
    assert settings.mcp_enabled is False
    assert _mcp_executor_and_schemas(settings, ToolRegistry(settings.tools_dir), ["package_search"]) is None
