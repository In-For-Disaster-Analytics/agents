"""Tests for the dso-geo MCP integration (spec 2026-06-30).

Covers the geo transform hard-block + token injection in MCPToolExecutor, the GeoSyncExecutor
submit→poll wrapper (terminal result, timeout not_ready, transform block), multi-server composite
routing with the no-overlap assertion, and the persona-nodes geo wiring (schema exclusion of
transforms, metadata-only persona exposure).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.tools.executor import (
    CompositeToolExecutor,
    GeoSyncExecutor,
    InProcessToolExecutor,
    MCPToolExecutor,
)
from app.tools.mcp_client import (
    GEO_PERSONA_METADATA_TOOLS,
    GEO_TRANSFORM_TOOLS,
    MCPClient,
)

# ── A fake geo MCP server (in-memory FastMCP) with a scripted execution lifecycle ──


@pytest.fixture()
def geo_client():
    fastmcp = pytest.importorskip("fastmcp")
    server = fastmcp.FastMCP("test-geo")
    state = {"polls": 0, "last_token": None, "submitted_args": None}

    @server.tool()
    def gdalinfo_extract(
        resource_id: str, include_stats: bool = True, tapis_token: str | None = None
    ) -> dict[str, Any]:
        """Extract metadata."""
        state["last_token"] = tapis_token
        return {"execution_id": "exec-1", "status": "SUBMITTED"}

    @server.tool()
    def reproject_raster(
        resource_id: str,
        target_crs: int,
        output_name: str,
        register_to_dataset: str | None = None,
        tapis_token: str | None = None,
    ) -> dict[str, Any]:
        """Reproject (transform — should never be called from the loop)."""
        state["submitted_args"] = {"resource_id": resource_id, "tapis_token": tapis_token}
        return {"execution_id": "exec-T", "status": "SUBMITTED"}

    @server.tool()
    def get_execution_status(execution_id: str, tapis_token: str | None = None) -> dict[str, Any]:
        """Poll: RUNNING for the first poll, then COMPLETE."""
        state["polls"] += 1
        if state["polls"] < 2:
            return {"execution_id": execution_id, "status": "RUNNING"}
        return {"execution_id": execution_id, "status": "COMPLETE", "result": {"metadata": {"crs": "EPSG:4326"}}}

    client = MCPClient(server, timeout=10.0)
    client.connect()
    client._state = state  # type: ignore[attr-defined]
    yield client
    client.close()


# ── MCPToolExecutor: geo transform block + token injection ─────────────────


def test_executor_blocks_geo_transforms(geo_client):
    ex = MCPToolExecutor(geo_client, token_arg="tapis_token", token_value="TOK")
    for name in GEO_TRANSFORM_TOOLS:
        out = ex.invoke(name, {"resource_id": "r", "target_crs": 4326, "output_name": "o.tif"})
        assert out["success"] is False
        assert out["error"]["code"] == "transform_blocked"
    assert geo_client._state["submitted_args"] is None  # never reached the server


def test_executor_injects_token_after_pop(geo_client):
    ex = MCPToolExecutor(geo_client, token_arg="tapis_token", token_value="SERVER_TOK")
    # model-supplied token must be dropped and replaced by the injected value
    ex.invoke("gdalinfo_extract", {"resource_id": "r", "tapis_token": "MODEL_TRIED"})
    assert geo_client._state["last_token"] == "SERVER_TOK"


# ── GeoSyncExecutor: submit → poll → terminal ───────────────────────────────


def test_geo_sync_polls_to_complete(geo_client):
    sync = GeoSyncExecutor(geo_client, token_value="TOK", poll_timeout=10.0, sleep=lambda s: None)
    out = sync.invoke("gdalinfo_extract", {"resource_id": "r"})
    assert out["success"] is True
    assert out["result"]["status"] == "COMPLETE"
    assert out["result"]["result"]["metadata"]["crs"] == "EPSG:4326"


def test_geo_sync_times_out_to_not_ready():
    fastmcp = pytest.importorskip("fastmcp")
    server = fastmcp.FastMCP("slow-geo")

    @server.tool()
    def gdalinfo_extract(resource_id: str, tapis_token: str | None = None) -> dict[str, Any]:
        return {"execution_id": "exec-slow", "status": "SUBMITTED"}

    @server.tool()
    def get_execution_status(execution_id: str, tapis_token: str | None = None) -> dict[str, Any]:
        return {"execution_id": execution_id, "status": "RUNNING"}  # never terminal

    client = MCPClient(server, timeout=10.0)
    client.connect()
    try:
        sync = GeoSyncExecutor(client, token_value="TOK", poll_timeout=0.05, sleep=lambda s: None)
        out = sync.invoke("gdalinfo_extract", {"resource_id": "r"})
        assert out["success"] is False
        assert out["error"]["code"] == "geo_not_ready"
        assert out["error"]["execution_id"] == "exec-slow"
    finally:
        client.close()


def test_geo_sync_blocks_transforms(geo_client):
    sync = GeoSyncExecutor(geo_client, token_value="TOK", poll_timeout=10.0, sleep=lambda s: None)
    out = sync.invoke("reproject_raster", {"resource_id": "r", "target_crs": 4326, "output_name": "o.tif"})
    assert out["error"]["code"] == "transform_blocked"


# ── multi-server composite routing + no-overlap ─────────────────────────────


def test_composite_routes_across_servers(geo_client):
    class _Fake(InProcessToolExecutor):
        def __init__(self):
            pass

        def invoke(self, name, args):
            return {"success": True, "tool": name, "result": "in_process"}

    geo_sync = GeoSyncExecutor(geo_client, token_value="TOK", poll_timeout=10.0, sleep=lambda s: None)
    composite = CompositeToolExecutor(_Fake(), {"gdalinfo_extract": geo_sync})
    assert composite.invoke("gdalinfo_extract", {"resource_id": "r"})["result"]["status"] == "COMPLETE"
    assert composite.invoke("file_read_text", {"path": "/x"})["result"] == "in_process"


def test_no_overlap_assertion():
    from app.agents.ckan_registration.persona_nodes import _assert_no_overlap
    from app.tools import ToolError

    _assert_no_overlap({"a", "b"}, {"c"}, "geo")  # no raise
    with pytest.raises(ToolError):
        _assert_no_overlap({"a", "b"}, {"b", "c"}, "geo")


def test_geo_persona_metadata_set_excludes_summary_and_status():
    assert "gdalinfo_extract" in GEO_PERSONA_METADATA_TOOLS
    assert "gdalinfo_summary" not in GEO_PERSONA_METADATA_TOOLS
    assert "get_execution_status" not in GEO_PERSONA_METADATA_TOOLS
    # transforms are entirely disjoint from the persona metadata set
    assert not (GEO_PERSONA_METADATA_TOOLS & GEO_TRANSFORM_TOOLS)
