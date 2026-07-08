"""Tool executors.

The engine depends only on the ``ToolExecutor`` protocol — ``invoke(name, args) -> envelope``.
``InProcessToolExecutor`` calls ``ToolRegistry`` directly (fast, no transport).
``MCPToolExecutor`` invokes tools on the standalone ``dso_ckan_mcp`` server over HTTP (spec
2026-06-29) — the same protocol, no engine changes. ``CompositeToolExecutor`` routes by tool
name so the author can use in-process file tools and MCP CKAN tools in one loop.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.tools.mcp_client import GEO_TRANSFORM_TOOLS, WRITE_TOOL_NAMES, MCPClient
from app.tools.registry import ToolRegistry
from app.tools.results import tool_error, tool_success


class ToolExecutor(Protocol):
    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        ...


class InProcessToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return self.registry.invoke(name, args)


class MCPToolExecutor:
    """Invoke one MCP server's tools, returning the standard ``results.py`` envelope.

    **Write/transform safety (Fork A / security Critical).** CKAN live writes (``dry_run`` not
    True) and ALL geo transform tools are hard-blocked here and never forwarded — the autonomous
    persona tool loop can only dry-run / read. Live writes and transforms are issued exclusively
    from the approval-gated graph node, which calls ``client.call_tool`` directly (bypassing this
    guard) after human approval.

    **Token injection (architect Risk 1).** ``token_arg`` names a tool argument into which
    ``token_value`` is injected **after** the defensive ``tapis_token`` pop, so a geo client can
    carry an arg-based token while the CKAN client (header auth) injects nothing. The model never
    supplies it: ``tapis_token`` is in ``MODEL_HIDDEN_ARGS`` and stripped from every schema.
    """

    def __init__(
        self,
        client: MCPClient,
        *,
        token_arg: str | None = None,
        token_value: str | None = None,
    ) -> None:
        self.client = client
        self.token_arg = token_arg
        self.token_value = token_value

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = dict(args or {})
        if name in WRITE_TOOL_NAMES and not bool(args.get("dry_run", False)):
            return tool_error(
                name,
                "live_write_blocked",
                "Live writes are not permitted from the tool loop; this surface is dry-run only. "
                "Writes require the human approval gate.",
            )
        if name in GEO_TRANSFORM_TOOLS:
            return tool_error(
                name,
                "transform_blocked",
                "Geo transforms spend compute and write CKAN resources; they are not callable "
                "from the tool loop. Transforms require the human approval gate.",
            )
        # Defense in depth: never let a model-supplied call carry a token; inject ours after.
        args.pop("tapis_token", None)
        if name in WRITE_TOOL_NAMES:
            args["dry_run"] = True
        if self.token_arg and self.token_value:
            args[self.token_arg] = self.token_value
        try:
            result = self.client.call_tool(name, args)
        except Exception as exc:  # noqa: BLE001 - surface as a structured tool error
            return tool_error(name, "mcp_error", str(exc))
        return tool_success(name, result)


class GeoSyncExecutor:
    """Adapt the geo server's async submit→poll model into one synchronous tool result.

    The geo metadata tool (``gdalinfo_extract``) returns ``{execution_id, status: SUBMITTED}``
    immediately. This executor submits, then polls ``get_execution_status`` with bounded backoff
    up to ``poll_timeout`` on the geo client's own loop thread, and returns the terminal result
    as a single envelope. On timeout it returns a structured ``geo_not_ready`` error telling the
    model to proceed without geo metadata — there is no false "resumable" claim (spec 2026-06-30).

    Transforms and other side-effectful geo tools are hard-blocked here too (defense in depth);
    only metadata reads should ever be routed to this executor.
    """

    _STATUS_TOOL = "get_execution_status"
    _TERMINAL = {"COMPLETE", "FAILED", "ERROR"}

    def __init__(
        self,
        client: MCPClient,
        *,
        token_value: str | None = None,
        poll_timeout: float = 90.0,
        sleep: Any = None,
    ) -> None:
        self.client = client
        self.token_value = token_value
        self.poll_timeout = poll_timeout
        import time as _time

        self._sleep = sleep or _time.sleep
        self._now = _time.monotonic

    def _with_token(self, args: dict[str, Any]) -> dict[str, Any]:
        args = dict(args or {})
        args.pop("tapis_token", None)
        if self.token_value:
            args["tapis_token"] = self.token_value
        return args

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in GEO_TRANSFORM_TOOLS or name in WRITE_TOOL_NAMES:
            return tool_error(name, "transform_blocked", "Side-effectful tools require the approval gate.")
        try:
            submitted = self.client.call_tool(name, self._with_token(args))
        except Exception as exc:  # noqa: BLE001
            return tool_error(name, "mcp_error", str(exc))
        if isinstance(submitted, dict) and submitted.get("error"):
            return tool_error(name, "geo_error", str(submitted["error"]))
        execution_id = submitted.get("execution_id") if isinstance(submitted, dict) else None
        if not execution_id:
            return tool_success(name, submitted)  # nothing to poll; return as-is

        deadline = self._now() + self.poll_timeout
        delay = 1.0
        while self._now() < deadline:
            try:
                status = self.client.call_tool(self._STATUS_TOOL, self._with_token({"execution_id": execution_id}))
            except Exception as exc:  # noqa: BLE001
                return tool_error(name, "mcp_error", str(exc))
            state = status.get("status") if isinstance(status, dict) else None
            if state in self._TERMINAL:
                return tool_success(name, status)
            self._sleep(min(delay, max(0.0, deadline - self._now())))
            delay = min(delay * 1.5, 8.0)
        return tool_error(
            name,
            "geo_not_ready",
            f"Geo execution {execution_id} did not finish within {self.poll_timeout:.0f}s. "
            "Proceed without geo metadata; do not retry in this turn.",
            execution_id=execution_id,
        )


class GeoTransformRunner:
    """Run a geo transform from the approval-gated node (NOT the persona loop).

    Unlike ``MCPToolExecutor``/``GeoSyncExecutor``, this DOES execute transforms — it is only
    constructed inside the human-approval-gated ``geo-apply`` node, after the user authorized the
    exact operation. It injects the Tapis token server-side, submits, polls to terminal up to
    ``poll_timeout``, and returns a scrubbed envelope. On timeout it returns the ``execution_id``
    with a RUNNING status so the caller can poll later via ``poll_status``. The token is injected
    only into outbound call args and never appears in the returned envelope.
    """

    _STATUS_TOOL = "get_execution_status"
    _TERMINAL = {"COMPLETE", "FAILED", "ERROR"}

    def __init__(
        self, client: MCPClient, *, token_value: str | None, poll_timeout: float = 120.0, sleep: Any = None
    ) -> None:
        self.client = client
        self.token_value = token_value
        self.poll_timeout = poll_timeout
        import time as _time

        self._sleep = sleep or _time.sleep
        self._now = _time.monotonic

    def _with_token(self, args: dict[str, Any]) -> dict[str, Any]:
        out = dict(args or {})
        out.pop("tapis_token", None)
        if self.token_value:
            out["tapis_token"] = self.token_value
        return out

    def run(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            submitted = self.client.call_tool(tool_name, self._with_token(args))
        except Exception as exc:  # noqa: BLE001
            return tool_error(tool_name, "mcp_error", str(exc))
        if isinstance(submitted, dict) and submitted.get("error"):
            return tool_error(tool_name, "geo_error", str(submitted["error"]))
        execution_id = submitted.get("execution_id") if isinstance(submitted, dict) else None
        if not execution_id:
            return tool_success(tool_name, submitted)
        terminal = self._poll(execution_id)
        if terminal is None:
            return tool_success(
                tool_name,
                {
                    "status": "RUNNING",
                    "execution_id": execution_id,
                    "note": "Transform still running; poll later with the transform-status action.",
                },
            )
        return tool_success(tool_name, terminal)

    def poll_status(self, execution_id: str) -> dict[str, Any]:
        try:
            status = self.client.call_tool(self._STATUS_TOOL, self._with_token({"execution_id": execution_id}))
        except Exception as exc:  # noqa: BLE001
            return tool_error("transform-status", "mcp_error", str(exc))
        return tool_success("transform-status", status)

    def _poll(self, execution_id: str) -> dict[str, Any] | None:
        deadline = self._now() + self.poll_timeout
        delay = 1.0
        while self._now() < deadline:
            try:
                status = self.client.call_tool(self._STATUS_TOOL, self._with_token({"execution_id": execution_id}))
            except Exception:  # noqa: BLE001
                return None
            if isinstance(status, dict) and status.get("status") in self._TERMINAL:
                return status
            self._sleep(min(delay, max(0.0, deadline - self._now())))
            delay = min(delay * 1.5, 8.0)
        return None


class CompositeToolExecutor:
    """Route tool calls by name across N MCP servers + the in-process registry.

    ``mcp_tools`` maps a tool name to the ``MCPToolExecutor`` that serves it (the flat map IS the
    multi-server registry — O(1) routing, trivially extensible). Anything not in the map goes
    in-process.
    """

    def __init__(self, in_process: InProcessToolExecutor, mcp_tools: dict[str, MCPToolExecutor]) -> None:
        self.in_process = in_process
        self.mcp_tools = dict(mcp_tools)

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        executor = self.mcp_tools.get(name)
        if executor is not None:
            return executor.invoke(name, args)
        return self.in_process.invoke(name, args)
