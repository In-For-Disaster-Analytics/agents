"""MCP client layer for the standalone ``dso_ckan_mcp`` server (spec 2026-06-29).

Connects the synchronous persona engine to the MCP server's CKAN tools/prompts over HTTP.

Design decisions (from the design review):
- **Async/sync bridge = owned background event-loop thread.** ``fastmcp.Client`` is async; the
  engine is sync and is dispatched inside FastAPI/LangGraph's event loop. ``asyncio.run()`` would
  raise "event loop is already running", so we run a dedicated loop in a daemon thread and submit
  coroutines via ``run_coroutine_threadsafe`` (B-async). The connection is persistent.
- **Auth is header-only (B2).** A shared secret (``Authorization: Bearer``) and an optional Tapis
  write token (``X-Tapis-Token``) are sent as HTTP headers — never as tool arguments, so they
  cannot enter model context, tool-call logs, or the checkpointer.
- **Schemas are normalized** from MCP ``inputSchema`` (Pydantic/JSON-Schema, may carry
  ``$ref``/``$defs``/``anyOf``) into OpenAI-function-safe parameter schemas, and model-hidden
  args (``tapis_token``; ``dry_run`` for write tools) are stripped from what the model sees.
- **Failures degrade to structured errors**, never unhandled exceptions in the engine path.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# CKAN write tools exposed by the MCP server (Track B). Live writes from these must never be
# reachable by the autonomous persona tool loop (Fork A); the executor hard-blocks dry_run=False.
WRITE_TOOL_NAMES: frozenset[str] = frozenset(
    {"schema_create_package", "schema_update_package", "schema_create_resource"}
)

# Geo transform tools (dso-geo): spend Abaco compute AND register new CKAN resources. Like CKAN
# writes, they are never reachable from the persona tool loop — the executor hard-blocks them and
# the schema-merge code excludes them unconditionally. Live runs go through the approval gate.
GEO_TRANSFORM_TOOLS: frozenset[str] = frozenset(
    {"reproject_raster", "convert_to_cog", "clip_raster", "build_overviews"}
)

# Tools never exposed to the autonomous persona loop (executor-blocked + schema-excluded).
PERSONA_BLOCKED_TOOLS: frozenset[str] = WRITE_TOOL_NAMES | GEO_TRANSFORM_TOOLS

# Geo tools personas MAY use during authoring (v1): single-execution metadata only.
# gdalinfo_summary (multi-execution fan-out) and get_execution_status (internal to the sync
# wrapper) are intentionally excluded from the persona surface.
GEO_PERSONA_METADATA_TOOLS: frozenset[str] = frozenset({"gdalinfo_extract"})

# Tool arguments that must never be shown to / set by the model.
MODEL_HIDDEN_ARGS: frozenset[str] = frozenset({"tapis_token"})


class MCPClientError(RuntimeError):
    """Raised for connection/transport problems the caller may want to handle explicitly."""


def get_mcp_prompt(settings: Any, name: str, **args: Any) -> str:
    """Thin accessor (O2): fetch a rendered MCP prompt by name from the configured server.

    Returns the prompt text, or "" if MCP is disabled/unreachable (callers degrade gracefully).
    """
    if not getattr(settings, "mcp_enabled", False):
        return ""
    try:
        client = get_shared_client(
            settings.mcp_server_url,
            shared_secret=getattr(settings, "mcp_shared_secret", "") or None,
            tapis_token=getattr(settings, "mcp_tapis_token", "") or None,
            timeout=getattr(settings, "mcp_timeout", 30.0),
        )
        return client.get_prompt(name, args)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP prompt %r unavailable: %s", name, exc)
        return ""


_SHARED_CLIENTS: dict[str, MCPClient] = {}
_SHARED_LOCK = threading.Lock()


def get_shared_client(
    url: str,
    *,
    shared_secret: str | None = None,
    tapis_token: str | None = None,
    timeout: float = 30.0,
) -> MCPClient:
    """Return a process-wide ``MCPClient`` for ``url`` (one loop thread + connection reused).

    Keyed by URL only — the auth headers are fixed at first construction for a given URL.
    """
    with _SHARED_LOCK:
        client = _SHARED_CLIENTS.get(url)
        if client is None:
            client = MCPClient(
                url, shared_secret=shared_secret, tapis_token=tapis_token, timeout=timeout
            )
            _SHARED_CLIENTS[url] = client
        return client


def _resolve_refs(schema: Any, defs: dict[str, Any], _seen: frozenset[str] = frozenset()) -> Any:
    """Recursively inline ``$ref`` against ``$defs``/``definitions`` (cycle-guarded)."""
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            key = ref.split("/")[-1]
            if key in _seen:  # cycle — leave a permissive object
                return {"type": "object"}
            target = defs.get(key)
            if target is not None:
                merged = _resolve_refs(target, defs, _seen | {key})
                # carry sibling keys (e.g. description) over the resolved target
                extra = {k: v for k, v in schema.items() if k != "$ref"}
                if isinstance(merged, dict):
                    return {**merged, **{k: _resolve_refs(v, defs, _seen) for k, v in extra.items()}}
                return merged
        return {k: _resolve_refs(v, defs, _seen) for k, v in schema.items() if k not in ("$ref",)}
    if isinstance(schema, list):
        return [_resolve_refs(v, defs, _seen) for v in schema]
    return schema


def _flatten_optional(prop: Any) -> Any:
    """Flatten Pydantic optional ``anyOf:[T, {type:null}]`` into a single nullable type."""
    if not isinstance(prop, dict):
        return prop
    variants = prop.get("anyOf") or prop.get("oneOf")
    if isinstance(variants, list):
        non_null = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
        has_null = any(isinstance(v, dict) and v.get("type") == "null" for v in variants)
        if len(non_null) == 1:
            base = dict(non_null[0])
            for k in ("description", "title", "default"):
                if k in prop and k not in base:
                    base[k] = prop[k]
            if has_null:
                base["nullable"] = True
            return _flatten_optional(base)
    return prop


_STRIP_KEYS = {"$schema", "$defs", "definitions", "title", "additionalProperties", "$id"}


def _clean_prop(prop: Any) -> Any:
    if isinstance(prop, dict):
        return {k: _clean_prop(v) for k, v in prop.items() if k not in _STRIP_KEYS}
    if isinstance(prop, list):
        return [_clean_prop(v) for v in prop]
    return prop


def normalize_input_schema(input_schema: dict[str, Any] | None, *, hidden_args: frozenset[str]) -> dict[str, Any]:
    """Convert an MCP ``inputSchema`` into an OpenAI-function-safe ``parameters`` object.

    Resolves ``$ref``/``$defs``, flattens optional ``anyOf``, strips JSON-Schema keywords the
    OpenAI tools API rejects, and drops model-hidden args.
    """
    schema = dict(input_schema or {})
    defs = schema.get("$defs") or schema.get("definitions") or {}
    resolved = _resolve_refs(schema, defs)
    properties = dict(resolved.get("properties") or {})
    required = [r for r in (resolved.get("required") or []) if r not in hidden_args]
    out_props: dict[str, Any] = {}
    for name, prop in properties.items():
        if name in hidden_args:
            continue
        out_props[name] = _clean_prop(_flatten_optional(prop))
    return {"type": "object", "properties": out_props, "required": required}


def _hidden_args_for(tool_name: str) -> frozenset[str]:
    hidden = set(MODEL_HIDDEN_ARGS)
    if tool_name in WRITE_TOOL_NAMES:
        # The model must never set dry_run; writes are gated elsewhere (Fork A).
        hidden.add("dry_run")
    return frozenset(hidden)


class MCPClient:
    """Synchronous facade over an async ``fastmcp.Client`` driven on a background loop thread.

    ``transport`` may be an HTTP URL (str), or any object ``fastmcp.Client`` accepts (a FastMCP
    server instance for in-memory tests). When a URL is given, a ``StreamableHttpTransport`` is
    built with the auth headers.
    """

    def __init__(
        self,
        transport: Any,
        *,
        shared_secret: str | None = None,
        tapis_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._timeout = timeout
        self._headers: dict[str, str] = {}
        if shared_secret:
            self._headers["Authorization"] = f"Bearer {shared_secret}"
        if tapis_token:
            self._headers["X-Tapis-Token"] = tapis_token
        self._raw_transport = transport
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, name="mcp-client-loop", daemon=True)
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def _submit(self, coro: Any, timeout: float | None = None) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout if timeout is not None else self._timeout + 5)

    def _build_transport(self) -> Any:
        if isinstance(self._raw_transport, str):
            from fastmcp.client.transports import StreamableHttpTransport

            return StreamableHttpTransport(self._raw_transport, headers=self._headers or None)
        return self._raw_transport

    async def _aconnect(self) -> None:
        if self._client is not None:
            return
        from fastmcp import Client

        client = Client(self._build_transport())
        await client.__aenter__()
        self._client = client

    def connect(self) -> None:
        self._submit(self._aconnect())

    def close(self) -> None:
        if self._client is not None and self._loop is not None:
            try:
                self._submit(self._client.__aexit__(None, None, None))
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.debug("MCP client teardown error", exc_info=True)
            self._client = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None
            self._thread = None

    # ── operations ─────────────────────────────────────────────────────────
    def ping(self) -> bool:
        """Liveness check used at startup. Returns True if the server responds."""
        try:
            self.connect()
            self._submit(self._client.ping())
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP server ping failed: %s", exc)
            return False

    def list_tools(self) -> list[Any]:
        self.connect()
        return self._submit(self._client.list_tools())

    def tool_names(self) -> list[str]:
        return [t.name for t in self.list_tools()]

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        allow = set(names) if names is not None else None
        out: list[dict[str, Any]] = []
        for tool in self.list_tools():
            if allow is not None and tool.name not in allow:
                continue
            params = normalize_input_schema(
                getattr(tool, "inputSchema", None), hidden_args=_hidden_args_for(tool.name)
            )
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": getattr(tool, "description", "") or tool.name,
                        "parameters": params,
                    },
                }
            )
        return out

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke an MCP tool and return a plain JSON-able result.

        Auth headers carry the token; per-call args carry only tool inputs.
        """
        self.connect()
        result = self._submit(self._client.call_tool(name, args or {}))
        return self._unwrap(result)

    def list_prompts(self) -> list[Any]:
        self.connect()
        return self._submit(self._client.list_prompts())

    def get_prompt(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Fetch a rendered prompt's text (joined message contents)."""
        self.connect()
        result = self._submit(self._client.get_prompt(name, args or {}))
        return self._prompt_text(result)

    # ── result shaping ───────────────────────────────────────────────────────
    @staticmethod
    def _unwrap(result: Any) -> Any:
        # fastmcp CallToolResult: prefer structured data, else text content.
        for attr in ("data", "structured_content", "structuredContent"):
            value = getattr(result, attr, None)
            if value is not None:
                return value
        content = getattr(result, "content", None)
        if isinstance(content, list):
            texts = [getattr(block, "text", None) for block in content]
            texts = [t for t in texts if t is not None]
            if len(texts) == 1:
                return texts[0]
            if texts:
                return texts
        return result if not hasattr(result, "model_dump") else result.model_dump()

    @staticmethod
    def _prompt_text(result: Any) -> str:
        messages = getattr(result, "messages", None) or []
        parts: list[str] = []
        for msg in messages:
            content = getattr(msg, "content", None)
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
            elif isinstance(content, str):
                parts.append(content)
        return "\n\n".join(parts)
