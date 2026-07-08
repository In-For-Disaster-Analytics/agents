# Integrate the running DSO CKAN MCP server as the agent's CKAN tool surface

## Status

Implemented (read/schema/validate + dry-run path); **Fork A apply-node re-pointing deferred** to
a follow-up increment.

_Reviewed 2026-06-29 by architect, skeptic, security-reviewer; findings resolved in
[Review synthesis](#review-synthesis-2026-06-29). Forks confirmed by user (2026-06-29): Fork A =
writes behind the approval gate only; Fork B = keep in-repo CKAN read tools as fallback._

### Implementation summary (2026-06-30)

Landed on branch `feat/integrate-dso-ckan-mcp`:

- **MCP server** (`dso_ckan_mcp`): `MCP_TRANSPORT=http` mode with a shared-secret bearer
  middleware bound to `127.0.0.1` by default (`config.py`, `server.py`). 4 HTTP-transport tests.
- **Agent**: `app/tools/mcp_client.py` (`MCPClient` — owned-loop sync bridge, header auth,
  `inputSchema`→OpenAI normalization, prompt accessor, shared-client factory);
  `MCPToolExecutor` + `CompositeToolExecutor` (`executor.py`) with the live-write hard-block and
  token scrub; `_tool_kwargs` wiring + `CKAN_MCP_*` settings, flag-gated with read-tool fallback.
  10 integration tests (in-memory FastMCP round-trip). `fastmcp>=3,<4` added to deps.
- **Both full suites green** (mcp-server 126 passed; agent 122 passed — 3 pre-existing
  unrelated `legacy_worker` failures untouched). READMEs + this spec updated.

**Deviation / deferred:** the in-repo CKAN read tools are **kept** (Fork B) rather than deleted.
The Fork A live-write path — re-pointing `make_safe_apply_node` from `LegacyCkanWorker` to the MCP
write tools behind the `interrupt()` gate — is a **tracked follow-up increment** (own diff +
security pass). Until then, MCP write tools exist but are unreachable from the persona loop (the
executor blocks them) and writes continue through the legacy gated worker. No persona has write
tools in its allow-list (default).

## Objective

Replace the agent's **in-repo CKAN tools** (`app/tools/catalog/ckan.yaml` +
`app/tools/handlers/ckan.py`) with calls to the **standalone `dso_ckan_mcp` MCP server**
(`ckan-docker/mcp-server`). The agent connects to a **running MCP server over HTTP** as an
MCP client; CKAN read/schema/validate tools, the gated write tools, and the server's prompt
templates become available to the tool-calling personas. File-extractor tools stay in-process.

This is the "MCP-client executor" path that `app/tools/executor.py` was explicitly designed
for ("When the MCP server lands (6b), an MCP-client executor can implement the same protocol
without engine changes"). It supersedes the in-repo `app/mcp/server.py` idea from the
[2026-06-26 tooling+MCP spec](2026-06-26-tooling-and-mcp.md): the server already exists as a
separate project, so we consume it rather than rebuild it.

## User need

- "We now have a running mcp-server that has the ckan tools and prompts so lets integrate that
  into our tools instead."
- Connect to a **running HTTP** MCP server (decisions below).
- **Replace** the in-repo CKAN tools with the MCP ones; keep file extractors in-process.
- **Pull in the MCP prompts** (4 read-only CKAN prompts) so personas can use them.
- **Allow the gated MCP write tools** (dry-run-first, token-gated) into the tool surface — a
  deliberate change from the current read-only-only posture (R4 in the prior spec).

## Decisions (user, 2026-06-29)

- **Transport = running HTTP server.** The MCP server runs with FastMCP HTTP transport; the
  agent connects as an HTTP MCP client. (Not stdio-subprocess.)
- **CKAN scope = replace.** Retire the in-repo CKAN handler/catalog; all CKAN tool calls route
  through the MCP server. File-extractor tools (`file_*`, `pdf_*`) remain in-process.
- **Prompts = pull in.** Expose the MCP server's 4 prompt templates to the agent.
- **Writes = allow gated MCP write tools.** Personas may reach the Track-B write tools
  (`schema_create_package`, `schema_update_package`, `schema_create_resource`), which are
  dry-run-first and token-gated on the server. This relaxes the prior "no write tools in the
  tool surface" rule and requires a security review (see Risks).

## Current system summary

- **Tool registry / executor** (`app/tools/`): `ToolRegistry` discovers `catalog/*.yaml`,
  resolves dotted handlers, and exposes `to_openai_tools()` (schema-gen) + `invoke()` (returns
  a `{success, tool, result|error}` envelope from `results.py`). The `ToolExecutor` protocol is
  just `invoke(name, args) -> envelope`; `InProcessToolExecutor` wraps the registry. A load-time
  guard refuses `read_only: false` tools (R4 write-guard).
- **CKAN tools today** (to be replaced): `ckan_package_show`, `ckan_package_search`,
  `ckan_organization_list`, `ckan_resolve_org`, `ckan_dry_run_diff` — thin wrappers over
  `app/agents/ckan_registration/ckan_client.py`.
- **Engine** (`app/personas/engine.py`): fully **synchronous**. `_author_tool_loop` runs the
  author with `author_tool_specs` (OpenAI schemas) and a `tool_executor`, capped at
  `max_tool_calls`. Gated behind `CKAN_PERSONA_TOOLS` + a persona `tools:` allow-list.
- **Consumer** (`app/agents/ckan_registration/persona_nodes.py:130` `_tool_kwargs`): builds the
  registry, `to_openai_tools(names=author.tools)`, and `InProcessToolExecutor(registry)`.
- **Personas**: only `domain_expert.md` declares `tools:` today, and only `file_*` + `pdf_*` —
  **no persona currently references any CKAN tool**, so replacing the CKAN backend is low-risk
  for existing behavior.
- **Prompts** (`app/prompts/`): file-based `PromptRegistry` (`<agent>/<name>.md`, `{{var}}`
  rendering). Unrelated to MCP prompts today.
- **The MCP server** (`dso_ckan_mcp`): FastMCP app, `mcp.run()` = **stdio only** at present.
  Tools: read (8) — `package_search, package_show, find_relevant_datasets, resource_show,
  organization_list, organization_show, group_list, get_capabilities`; schema (2) —
  `list_dataset_types, describe_dataset_schema`; validation (1) — `validate_metadata`; write (3,
  gated) — `schema_create_package, schema_update_package, schema_create_resource`. Prompts (4) —
  `analyze_dataset, find_by_variable, recent_datasets, describe_org_holdings`. Plus resources.
  Tool names have **no `ckan_` prefix** (naming-mapping concern below).

## Proposed design

### 1. MCP client layer (`app/tools/mcp_client.py`, new)

A thin wrapper around `fastmcp.Client` (HTTP) that:

- Connects to `settings.mcp_server_url` (e.g. `http://localhost:8100/mcp`).
- `list_tools()` → MCP tool definitions, converted to the agent's OpenAI-tool schema shape
  (reuse the same `{type:"function", function:{name, description, parameters}}` form
  `ToolRegistry.to_openai_tool()` produces).
- `call_tool(name, args)` → raw MCP result.
- `list_prompts()` / `get_prompt(name, args)` → for prompt integration.

**Async/sync bridge.** `fastmcp.Client` is async; the engine is sync. The client layer exposes
**synchronous** methods that drive the async client via a dedicated background event loop
(a module-level loop thread, or `anyio.from_thread`/`asyncio.run` per call with a short-lived
connection). A persistent connection is preferred for latency; the bridge detail is an
implementation choice, but the public surface the engine sees is synchronous and matches the
existing `ToolExecutor` protocol. Connection failures degrade to a structured `tool_error`
envelope (the author loop already tolerates tool errors), never an unhandled exception.

### 2. `MCPToolExecutor` (`app/tools/executor.py`)

Implements the existing `ToolExecutor` protocol:

```python
class MCPToolExecutor:
    def invoke(self, name, args) -> dict:   # returns tool_success/tool_error envelope
        ...
```

It calls the MCP client's `call_tool` and wraps the result in the same `results.py` envelope so
the engine is unchanged.

### 3. Composite routing (`CompositeToolExecutor`, `app/tools/executor.py`)

The author can call **both** file tools (in-process) and CKAN tools (MCP). A composite executor
routes by tool name:

- Names served by MCP (discovered via `list_tools()`) → `MCPToolExecutor`.
- Everything else (`file_*`, `pdf_*`) → `InProcessToolExecutor`.

Schema generation likewise **merges** MCP tool schemas + the in-process file-tool schemas, then
filters by the persona's `tools:` allow-list. `_tool_kwargs` in `persona_nodes.py` is updated to
build this composite and the merged schema list.

### 4. Tool naming

MCP CKAN tools are unprefixed (`package_search`), while the in-repo ones were `ckan_*`. Since
**no persona currently lists a CKAN tool**, we adopt the **MCP names as-is** (no `ckan_`
prefix) — simplest, and matches the MCP server's own prompts which reference `package_show`
etc. by name. Personas opt in by adding MCP tool names to their `tools:` allow-list. (Decision
recorded; alternative — re-prefix in the client — rejected as needless remapping.)

The in-repo `ckan_dry_run_diff` has **no MCP equivalent**; its role (preview a write) is now
served by the MCP write tools' `dry_run=True` mode + `validate_metadata`. Removing it is
acceptable because nothing references it in a persona allow-list.

### 5. In-repo CKAN tools: keep read tools as fallback (Fork B)

**Do not delete** the in-repo CKAN tools. MCP is the default CKAN path; the in-repo CKAN
**read** tools (`app/tools/catalog/ckan.yaml` + `handlers/ckan.py`) are retained as a
**disabled-by-default fallback** so `CKAN_MCP_ENABLED=false` or an MCP outage degrades
gracefully rather than leaving the agent with no CKAN tools. The composite executor/schema
source selects MCP when enabled and reachable, else the in-repo read tools. `app/agents/
ckan_registration/ckan_client.py` stays (also used by the gated graph write path).

### 6. MCP prompts integration

Add a small `app/tools/mcp_prompts.py` (or extend the MCP client) that lists/fetches the MCP
server's prompts and exposes them to the agent. Minimal first cut: a helper the persona layer
can call to fetch a rendered prompt by name+args. (Exact wiring — whether these surface as
PromptRegistry entries or a separate MCP-prompt accessor — is an open question, O2.)

### 7. MCP server: add HTTP transport (`ckan-docker/mcp-server`)

The server currently only runs stdio. Add an HTTP run mode without breaking stdio:

- `main()` reads `MCP_TRANSPORT` (`stdio` default | `http`) and, for HTTP, `MCP_HTTP_HOST` /
  `MCP_HTTP_PORT`, then calls `mcp.run(transport="http", host=..., port=...)`.
- Document the HTTP run command in the MCP server README.

### 8. Settings (`app/settings.py`)

New env-backed settings:

| Setting | Env | Default | Purpose |
|---|---|---|---|
| `mcp_enabled` | `CKAN_MCP_ENABLED` | `false` | Master flag: route CKAN tools via MCP |
| `mcp_server_url` | `CKAN_MCP_URL` | `http://localhost:8100/mcp` | HTTP MCP endpoint |
| `mcp_timeout` | `CKAN_MCP_TIMEOUT` | `30` | Per-call timeout (s) |
| `mcp_tapis_token` | `CKAN_MCP_TAPIS_TOKEN` | *(unset)* | Optional per-call write token forwarded to gated write tools |

`tools_dir` and the existing `persona_tools_enabled`/`max_tool_calls` are unchanged.

## Files likely affected

| Path | Change |
|---|---|
| `app/tools/mcp_client.py` (new) | FastMCP HTTP client + sync bridge + schema/prompt conversion |
| `app/tools/executor.py` | add `MCPToolExecutor` + `CompositeToolExecutor` |
| `app/tools/mcp_prompts.py` (new, optional) | MCP prompt accessor |
| `app/agents/ckan_registration/persona_nodes.py` | `_tool_kwargs` builds composite + merged schemas |
| `app/tools/catalog/ckan.yaml`, `app/tools/handlers/ckan.py` | **kept** as disabled-by-default fallback (Fork B) |
| `app/agents/ckan_registration/{graph,nodes}.py` (approval/apply) | re-point the gated write node at MCP write tools (Fork A) |
| `app/settings.py` | new `mcp_*` settings |
| `pyproject.toml` / `environment.yml` | add `fastmcp` (client) dependency |
| `ckan-docker/mcp-server/src/dso_ckan_mcp/server.py` | HTTP transport mode in `main()` |
| `ckan-docker/mcp-server/README.md`, agent `README.md` | run + integration docs |
| tests (both repos) | client (mocked), composite routing, schema merge, write-gating, HTTP smoke |

## API / schema changes

- No new agent HTTP endpoints. The agent gains an **outbound** MCP/HTTP dependency.
- Persona frontmatter: `tools:` may now list MCP tool names (unprefixed CKAN names).
- Engine signatures unchanged (composite executor satisfies the existing `ToolExecutor`).
- MCP server entrypoint gains `MCP_TRANSPORT` / `MCP_HTTP_HOST` / `MCP_HTTP_PORT` env support.

## Data flow

1. `persona` node builds evidence (unchanged) + a `CompositeToolExecutor` and merged tool
   schemas filtered by the author's allow-list (only when `CKAN_MCP_ENABLED` + `CKAN_PERSONA_TOOLS`).
2. Author calls tools → composite routes `file_*` in-process, CKAN names to the MCP HTTP server
   → results wrapped in the standard envelope → fed back to the author → final JSON.
3. **Writes**: a write tool call goes to MCP with `dry_run=True` by default; the server returns a
   preview. A live write (`dry_run=False`) is only issued after the existing approval gate and
   requires a token (`CKAN_MCP_TAPIS_TOKEN` or per-call). The agent must never let the model flip
   `dry_run=False` without passing through the approval gate (see Risks R2/R4).
4. The legacy REGISTER-gated graph write path is unaffected by this change.

## Risks and tradeoffs

- **R1 — write tools now in the model-callable surface (highest).** Relaxing the read-only
  posture means the author/personas can invoke `schema_create_*`/`schema_update_*`. Mitigations:
  default `dry_run=True`; require the existing human approval gate before any `dry_run=False`;
  token must be configured server- or call-side; the model must never be able to both decide to
  write *and* supply the live-write flag in one uncontrolled step. **Security review required
  before implementation.**
- **R2 — prompt injection via tool/prompt results escalating to a write.** File/CKAN content
  enters context; a malicious payload could try to induce a live write. Keep tool results
  delimited as data; gate writes on the human approval token, not on model text.
- **R3 — network dependency + latency.** The agent now depends on a reachable HTTP MCP server.
  Mitigations: `mcp_enabled` flag (off → current in-process behavior, but CKAN tools are gone —
  see open question O1), per-call timeout, connection failure → structured tool error, cap
  `max_tool_calls`.
- **R4 — server-side guards are the real backstop.** The MCP server enforces dry-run-first,
  token-gating, and `MCP_ALLOW_PROD_WRITES`. The agent must not assume client-side checks
  suffice. Assert in tests that a live write without a token/approval is refused.
- **R5 — async/sync bridge correctness.** Driving an async client from the sync engine risks
  event-loop reentrancy/leaks. Mitigation: a single owned loop or per-call `asyncio.run` with a
  short-lived connection; cover with tests.
- **R6 — tool-name collisions.** If an MCP tool name ever equals an in-process tool name, the
  composite routing is ambiguous. Mitigation: assert no overlap at startup.

## Alternatives considered

1. **stdio subprocess instead of HTTP** — simpler lifecycle, no listener, but the user chose a
   running HTTP server. (Rejected per decision.)
2. **Keep in-repo CKAN tools as a fallback** — more resilient if MCP is down, but the user chose
   replace. (Rejected per decision; revisit if R3 bites — see O1.)
3. **Re-prefix MCP CKAN tools as `ckan_*` in the client** — preserves old names, but no persona
   uses them and the MCP prompts reference bare names; needless remapping. (Rejected.)
4. **Build the in-repo `app/mcp/server.py` from the old spec** — duplicates the now-existing
   standalone server. (Superseded.)

## Test plan

- **MCP client** (mocked transport): `list_tools` → schema conversion shape; `call_tool` →
  envelope; connection failure → `tool_error`; sync-bridge does not leak loops.
- **Composite executor**: routes `file_*` in-process, CKAN names to MCP; merged schema list is
  the union, filtered by allow-list; **no-overlap assertion**.
- **persona_nodes `_tool_kwargs`**: with `CKAN_MCP_ENABLED` off → unchanged/no-tools path;
  on → composite built, schemas merged.
- **Write gating (R1/R4)**: a write tool defaults to `dry_run=True`; a `dry_run=False` call
  without approval/token is refused; assert the model cannot emit the live-write flag without
  the gate. (Mirror the MCP server's own write-gating tests.)
- **MCP server HTTP mode**: `MCP_TRANSPORT=http` boots and serves `list_tools` over HTTP (smoke).
- **Regression**: existing engine tests (no-tools path) stay byte-identical green.

## Documentation plan

- Agent `README.md`: how to point the agent at a running MCP server (`CKAN_MCP_*` env), what
  tools/prompts become available, the write-approval flow.
- MCP server `README.md`: how to run in HTTP mode.
- Update the [2026-06-26 tooling+MCP spec](2026-06-26-tooling-and-mcp.md) status to note the
  in-repo `app/mcp/server.py` plan is superseded by consuming the standalone server.

## Rollout / rollback plan

- Land additively behind `CKAN_MCP_ENABLED=false` default. With the flag off, CKAN tools are
  simply absent from the surface (since the in-repo ones are removed) — see O1 for whether "off"
  should instead retain a fallback.
- Add HTTP mode to the MCP server first (independent, testable).
- Then the client + composite executor + `_tool_kwargs` wiring.
- Flip `CKAN_MCP_ENABLED=true` in environments with a reachable server after the write-gating
  and smoke tests pass.
- Rollback: set `CKAN_MCP_ENABLED=false` (and, if needed, restore the deleted CKAN catalog/handler
  from git history).

## Review synthesis (2026-06-29)

Three reviewers ran in parallel. Verdicts: architect = revise; skeptic = revise; security =
**block** pending Critical/High items. Where the fix is unambiguous it is **baked into the spec
below**; two items are genuine forks that **need user confirmation** before this spec is Approved.

### Blocking findings → resolutions (baked in, no further input needed)

- **B1 — Live writes must be unreachable from the autonomous persona tool loop (security
  Critical #1; architect R1; skeptic Rank 3).** The existing REGISTER approval gate guards only
  the legacy graph write path, **not** persona tool calls. As designed, the author LLM could emit
  `schema_create_package(dry_run=False)` and the executor would forward it. **Resolution:**
  `MCPToolExecutor`/`CompositeToolExecutor` **hard-blocks** any write tool with `dry_run` not
  `True` and returns a `tool_error`; the model-visible tool schema for write tools **omits/forces
  `dry_run=True`** so the model cannot supply a live-write flag. Live writes are issued **only**
  from a graph node that has passed a LangGraph `interrupt()` human-approval gate (the existing
  `make_approval_node`/`make_safe_apply_node` pattern, re-pointed at the MCP write tools). See
  Fork A — this is the recommended write-surface model.
- **B2 — The write token must never be a model-visible tool argument (security Critical #2).**
  The MCP write tools expose `tapis_token` as a parameter; if forwarded in `args` it lands in LLM
  context, tool-call logs, and the SQLite checkpointer. **Resolution:** inject auth as an
  **HTTP header** from the MCP client layer (resolves O3); **strip `tapis_token` from the OpenAI
  tool schema** handed to the model; never let a token enter message history/checkpoints. Verify
  `fastmcp.Client` HTTP transport supports custom request headers (fallback: server-side
  `CKAN_API_TOKEN`, agent forwards no token).
- **B3 — The MCP HTTP endpoint has no auth by default (security High #3).** FastMCP HTTP binds
  and serves unauthenticated; with a `CKAN_API_TOKEN` set server-side, any process reaching the
  port gets ambient CKAN write access. **Resolution:** default `MCP_HTTP_HOST=127.0.0.1` (not
  `0.0.0.0`); add a **shared-secret bearer-token middleware** the agent supplies on every call;
  document the `CKAN_API_TOKEN` ambient-privilege risk; no public exposure without a fronting
  proxy.

### High/medium findings → resolutions (baked in)

- **Async/sync bridge = owned background event-loop thread** driving the async `fastmcp.Client`
  via `asyncio.run_coroutine_threadsafe`, with a persistent connection. **`asyncio.run()` is
  explicitly ruled out** — it raises `RuntimeError: event loop is already running` inside the
  FastAPI/LangGraph-dispatched thread. (architect Rank 2; skeptic Rank 1.) Lifecycle: start
  lazily/at app startup, stop at shutdown; connection failure → restart loop + `tool_error`.
- **MCP `inputSchema` → OpenAI function-schema normalization is a named sub-task** of
  `mcp_client.py`: resolve `$ref`/`$defs`, flatten Pydantic optional `anyOf:[T,null]`, strip
  `title`/`$schema`/unknown keywords. Tested. (architect Rank 3.)
- **Prompt-injection hardening:** wrap every CKAN tool result in a clearly labeled
  data-only envelope before re-entering model context; never let tool text carry control.
  (security High #4.)
- **Fail-fast startup health check** against the MCP server; do not silently degrade per request.
  (skeptic Rank 4.) Note the two-repo deployment ordering (MCP server up before agent).
- **Default allow-lists: no persona gets write tools** — explicit per-persona opt-in only,
  documented. (security Medium #8; resolves O4.) Author gets read/schema/validate only.
- **Token expiry:** Tapis JWTs are short-lived; document refresh, and surface a clear error
  (not a silent dry-run-then-fail) when a live write is attempted with an expired token.
  (skeptic Rank 5.)
- **Docs:** `MCP_ALLOW_PROD_WRITES` default + the agent-`CKAN_URL` / MCP-server-`CKAN_URL`
  divergence must be documented together; startup banner shows effective write-gate status.
  Private-IP hosts classify as production (intentional, but surprising). (security High #5/Med #6.)

### Forks needing user confirmation

- **Fork A — write-surface model.** Recommended (and required for security sign-off): MCP write
  tools are **not** callable by the autonomous author loop; they are wired behind the existing
  human-approval `interrupt()` gate (replacing the legacy worker as the write mechanism). This
  still "uses the gated MCP write tools" — just from the approval node, not the LLM loop.
  Alternative (more work, higher risk): build a new in-loop approval interrupt so the author can
  propose a live write mid-conversation.
- **Fork B — flag-off / rollback posture.** All three reviewers flag that "replace + delete
  in-repo CKAN tools" makes `CKAN_MCP_ENABLED=false` and any MCP outage a *no-CKAN-tools* state,
  so it is not a real rollback. Recommended: **keep the in-repo CKAN read tools as a
  disabled-by-default fallback** for graceful degradation, while MCP is the default path.
  Alternative: hard replace (delete now) and treat the MCP server as a hard runtime dependency
  with git-restore as the only rollback.

## Open questions

_All resolved by the 2026-06-29 review + user forks:_

1. **O1 — flag-off behavior — RESOLVED (Fork B):** keep in-repo CKAN read tools as a
   disabled-by-default fallback; flag-off / MCP-outage degrades to those rather than no tools.
2. **O2 — prompt surfacing — RESOLVED:** a thin MCP-prompt accessor (parameterised server
   prompts), not `PromptRegistry` file entries.
3. **O3 — write token source — RESOLVED (B2):** inject as an HTTP header from the MCP client
   layer; never a model-visible tool arg; never persisted to message history/checkpoints.
4. **O4 — which personas get write tools — RESOLVED:** none by default; writes are not in any
   persona tool surface (Fork A — writes flow only through the approval-gated graph node).

## Decisions

- 2026-06-29: Transport = running HTTP server; pull in MCP prompts (thin accessor); allow gated
  MCP write tools. (User, AskUserQuestion.)
- 2026-06-29: **Fork A — writes behind the approval gate only.** MCP write tools are NOT in any
  persona/author tool surface; live writes (`dry_run=False`) are issued solely from the existing
  REGISTER human-approval `interrupt()` node, re-pointed at the MCP write tools. The executor
  hard-blocks `dry_run=False` from the tool loop and the write-tool schema is scrubbed of any
  live-write/token arg. (User, confirming security Critical #1.)
- 2026-06-29: **Fork B — keep in-repo CKAN read tools as a disabled-by-default fallback** rather
  than deleting them; MCP is the default path, with graceful degradation on flag-off/outage.
  (User, confirming reviewer consensus.)
- 2026-06-29: Async/sync bridge = owned background event-loop thread (`run_coroutine_threadsafe`);
  `asyncio.run()` ruled out. Token via HTTP header (B2). MCP HTTP endpoint binds `127.0.0.1` +
  shared-secret middleware (B3). MCP `inputSchema`→OpenAI normalization is an explicit sub-task.
  (Review resolutions, baked in.)

## User feedback / decisions

- 2026-06-29: User asked to integrate the running `dso_ckan_mcp` server's CKAN tools and prompts
  into the agent's tools instead of the in-repo implementation, on a new branch
  (`feat/integrate-dso-ckan-mcp`). Forks answered as in Decisions above.
- _Awaiting user review of this spec before implementation._
