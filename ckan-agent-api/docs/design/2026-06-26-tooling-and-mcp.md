# Tooling Layer + MCP Server (tool-calling personas)

## Status

Implementing — the in-repo `app/mcp/server.py` plan (section 2) is **superseded** by
[2026-06-29-integrate-dso-ckan-mcp.md](2026-06-29-integrate-dso-ckan-mcp.md): the MCP server now
exists as the standalone `dso_ckan_mcp` project and the agent consumes it over HTTP rather than
hosting its own. The tool registry / catalog / handlers (increment 6a) and the engine tool-loop
(6c) described here remain in force.

_R1 probe PASSED (2026-06-26): `Meta-Llama-3.3-70B-Instruct` on the tejas endpoint
volunteered a valid tool call with `tool_choice="auto"`. Tool-calling personas are
viable; the engine tool-loop is approved to proceed (behind a flag, after the additive
layer). Rollout: 6a registry+catalog+handlers (additive) → 6b MCP server → 6c engine
tool-loop (after O1)._

_Increment 6a (DONE 2026-06-26): `app/tools/` — `ToolRegistry` (YAML catalog + dotted
handler, `safe_load`, path-confined, loud validation, **R4 write-guard: read_only=false is
refused at load**), `results.py` envelope, `handlers/ckan.py` (5 read-only CKAN tools) +
`handlers/files.py` (9 extractor tools, each `validate_readable_file`-checked), catalog
`ckan.yaml`/`files.yaml`, settings (`tools_dir`, `persona_tools_enabled`, `max_tool_calls`).
`to_openai_tools()` schema-gen ready for the engine. 9 tool tests; 69 passing overall.
`mcp` SDK is NOT yet installed in the env (needed for 6b)._

_O1 resolved (user 2026-06-26): in-process executor (default). Increment 6c (DONE): the engine
gained a tool-calling author loop — `_author_tool_loop` runs the author with read-only tools via a
`ToolExecutor` (default `InProcessToolExecutor` → `ToolRegistry.invoke`), feeds tool results back,
caps at `max_tool_calls` with a forced final JSON turn. `app/llm.invoke_chat_tools` is the OpenAI
tools turn; `Persona` gained a `tools:` allow-list (author seed lists file_* + ckan_package_search +
ckan_dry_run_diff); the persona node wires tools only when `CKAN_PERSONA_TOOLS=1` and the author
declares tools. **No-tools path is byte-identical (existing engine tests green).** 6 new tests; 75
passing. Evaluators get no tools yet (O2 default). 6b MCP server still pending (`mcp` dep)._

## Objective

Introduce a first-class **tool layer**: CKAN operations and the file extractors become
registered tools, discoverable and easy to add (parallel to the persona/schema registries),
exposed through an **MCP server**, and callable by the personas, which become **tool-calling
agents**. CKAN **writes stay gated** in the existing graph path; only read + dry-run are tools.

## User need

- "Set up the CKAN parts as tools as an MCP server."
- "Our other tools [file extractors] should be outlined as well [as] persona so it's easy to add tools."
- Personas should be able to *call* tools while reasoning (decision below), not just receive evidence.

## Decisions (user, 2026-06-26)

- **MCP consumer = internal only** (the agent uses the tools; not exposed to external clients yet).
- **Personas become tool-calling now.**
- **CKAN exposure = read + dry-run only**; create/update/resource-upload/apply remain behind the
  existing `REGISTER`-gated graph path (`approval`/`apply` nodes + legacy worker).
- **Tool format = YAML catalog + Python handler**, discovered by a `ToolRegistry` that mirrors
  `PersonaRegistry`. MCP tool schemas are generated from the catalog.

## Current system summary (reusable pieces)

- `app/agents/ckan_registration/ckan_client.py` — `package_show`, `package_search`,
  `organization_show/list`, `resolve_organization_id`, `resource_upload`, `action_get/post`.
  (Read primitives ready; writes exist but are out of scope for tools.)
- `app/files/` — extractors (`text, tabular, json_data, geojson, pdf, image, spatial, archive`)
  + `safety.validate_readable_file` (size/sensitive guard) + `analyze_path` dispatcher.
- `app/personas/registry.py` — the pattern to mirror for `ToolRegistry`.
- `archive/basic-ckan-agent/.../files/tool_catalog.yaml` — a clean, proven catalog schema
  (`name, summary, description, args, returns, use_when, limitations, safety`) to revive.
- `app/personas/engine.py` — currently a single-shot author→evaluators loop; gains a tool-use loop.
- No MCP dependency yet (will add the official `mcp` Python SDK).

## Proposed design

### 1. `app/tools/` — the tool registry (YAML catalog + handlers)

```
app/tools/
  catalog/
    ckan_package_show.yaml      file_read_text.yaml
    ckan_package_search.yaml    file_profile_csv.yaml
    ckan_organization_list.yaml file_profile_json.yaml
    ckan_dry_run_diff.yaml      file_profile_geojson.yaml
    ckan_resolve_org.yaml       file_extract_pdf_text.yaml
                                file_inspect_image.yaml / _zip.yaml / _raster.yaml / _shapefile_zip.yaml
  handlers/
    ckan.py        # thin wrappers over ckan_client (read-only header)
    files.py       # wrappers over app/files extractors, each safety-checked
  registry.py      # ToolRegistry — discover, validate, resolve handler, invoke, to_mcp_schema
```

A catalog entry (revived format):

```yaml
name: ckan_package_search
summary: Search CKAN datasets by free text.
description: Returns up to N matching datasets (name, title, org) for disambiguation.
category: ckan          # ckan | file
read_only: true
args:
  query:  {type: string, required: true}
  rows:   {type: integer, default: 10}
returns: {type: array, description: "Matching dataset summaries."}
use_when:
  - You need to check whether a dataset already exists before proposing a new one.
safety:
  - Read-only CKAN call; no writes.
handler: app.tools.handlers.ckan:package_search
```

`ToolRegistry` (mirrors `PersonaRegistry`): discovers `catalog/*.yaml`, validates required keys,
**`yaml.safe_load` only**, resolves the dotted `handler`, and exposes:
- `list_tools()` / `get(name)`
- `to_openai_tools()` / `to_mcp_tools()` — schema generation from the catalog
- `invoke(name, args) -> dict` — validates args against the catalog, calls the handler, returns a
  structured `{success, tool, result|error}` envelope (revive `results.py`).

**Tools shipped (all read-only):** CKAN — `package_show`, `package_search`, `organization_list`,
`resolve_org`, `dry_run_diff` (compare a desired payload vs. live CKAN). File — the nine extractor
tools, each wrapped with `validate_readable_file`. **No write tools are registered** (fork #3).

### 2. `app/mcp/server.py` — MCP server (internal)

A server built on the official `mcp` Python SDK (FastMCP) that registers every `ToolRegistry`
tool as an MCP tool, schemas generated from the catalog. Run via `python -m app.mcp.server`
(stdio) — wired for our own agent's use now; external exposure is a later flip (no rework, since
it's a real MCP server). Because only read/dry-run tools are registered, the MCP surface cannot
write to CKAN.

### 3. Tool execution: a `ToolExecutor` abstraction (the key design call)

"Internal-only MCP" + "tool-calling personas" can be built two ways. Recommendation:

- **In-process executor (default):** the engine calls `ToolRegistry.invoke()` directly. Fast, no
  transport, no second process in the request path.
- **MCP-client executor (optional):** the engine connects to `app/mcp/server` as an MCP client and
  invokes over the wire — for process isolation or when the same tools are served externally.

Both implement one `ToolExecutor` interface; the engine depends on the interface. **Default =
in-process**; the MCP server is the *exposure surface* and the alternate executor. This honors
"build an MCP server" and "internal consumer" without paying MCP round-trip cost on every tool call.
*(Open question O1 — confirm this vs. forcing MCP-wire internally.)*

### 4. Tool-calling persona engine

`run_persona_metadata_loop` gains a tool-use loop for the **author** (and optionally evaluators):

1. Build the chat request with the catalog's tool schemas (filtered by persona — see below).
2. If the model returns tool calls, execute them via the `ToolExecutor`, append results as tool
   messages, and re-call — up to a `max_tool_calls` cap.
3. When the model returns final content, parse the candidate metadata JSON as today.

- Persona frontmatter gains an optional `tools:` allow-list (e.g. author → file_* + ckan_package_search;
  evaluators → none or read-only). A persona with no `tools` key behaves exactly as today (no tools),
  so existing behavior and tests are preserved.
- Requires the LLM endpoint to support OpenAI-style tool calling (**risk R1**).

### Files likely affected

| Path | Change |
|---|---|
| `app/tools/registry.py`, `catalog/*.yaml`, `handlers/{ckan,files}.py` (new) | Tool registry + catalog + handlers |
| `app/tools/results.py` (new, revived) | `{success, tool, result/error}` envelope |
| `app/mcp/server.py` (new) | MCP server over the registry |
| `app/personas/engine.py` | Tool-use loop + `ToolExecutor` param; default no-tools path unchanged |
| `app/personas/registry.py` + seed `*.md` | Optional `tools:` allow-list in frontmatter |
| `app/settings.py` | `tools_dir`, `mcp_*`, `max_tool_calls`, executor mode |
| `pyproject.toml` / `environment.yml` | add `mcp` SDK |
| tests | registry, handlers (mocked CKAN), MCP schema-gen, engine tool-loop (mocked) |

### API/schema changes

- No new HTTP endpoints. MCP is a separate (stdio) server process.
- Persona frontmatter: optional `tools: [name, ...]`.
- Engine signature: optional `tool_executor` + `max_tool_calls`.

### Data flow (tool-calling author)

1. `persona` node builds evidence (unchanged) + a `ToolExecutor`.
2. Engine calls author with tool schemas → author may call `file_profile_csv`, `ckan_package_search`,
   etc. → executor runs them (read-only/safety-checked) → results fed back → author finalizes JSON.
3. Evaluators run (optionally with read-only tools). Clarification/propose unchanged.
4. Writes still only happen later via the `REGISTER`-gated `apply` path.

## Risks and tradeoffs

- **R1 (highest): model tool-calling support.** `Meta-Llama-3.3-70B-Instruct` on the tejas endpoint
  may not reliably do OpenAI-style function calling. Mitigation: capability-probe the endpoint first;
  if unreliable, fall back to the current evidence-up-front author and expose tools only via MCP /
  to evaluators. **Validate before committing the engine rewrite.**
- **R2 prompt-injection via tool results / file contents.** Tool outputs (file text, CKAN notes) enter
  the model context; a malicious file could carry instructions. Keep tool results clearly delimited as
  data; never let a tool emit the approval token; writes remain out of the tool surface.
- **R3 latency/cost.** Tool loops add round-trips. Cap `max_tool_calls`; default in-process executor.
- **R4 MCP surface.** Even internal, the server must register read-only tools only and never the
  CKAN write/apply handlers; assert this in a test.
- **R5 scope creep into the engine.** The tool loop is a real rewrite; gate it behind a flag and keep
  the no-tools path identical so the 60 passing tests stay green.

## Alternatives considered

1. **Personas stay evidence-up-front; MCP exposes tools externally only.** Lower risk, but the user
   explicitly chose tool-calling personas. (Rejected per decision.)
2. **Force MCP-wire calls internally.** Cleanest conceptually, but adds a process + transport to every
   tool call for no current benefit (internal-only). Deferred via the `ToolExecutor` abstraction (O1).
3. **Decorator/`@tool` registry.** More code-native but less declarative; YAML catalog chosen for
   "easy to add" + auto MCP schema-gen.

## Test plan

- ToolRegistry: catalog discovery/validation, `safe_load`, handler resolution, arg validation,
  `to_mcp_tools()`/`to_openai_tools()` schema shape, **assert no write tool is registered (R4)**.
- Handlers: CKAN read against a mocked `CkanClient`; file handlers reuse the extractor fixtures +
  safety refusals.
- MCP server: tools list matches the registry; a sample `invoke` round-trips.
- Engine tool-loop: with a fake tool-calling chat fn — author issues a tool call, gets a result,
  finalizes; `max_tool_calls` cap; **no-tools persona path byte-identical to today** (regression).
- Capability probe (R1): a small script/test hitting the real endpoint with one tool call.

## Documentation plan

- README: how to run the MCP server, how to add a tool (catalog YAML + handler), the `tools:` allow-list.
- A short "tools" section alongside the personas/schemas docs.

## Rollout / rollback plan

- Land the registry + handlers + MCP server first (additive, no engine change) — usable/testable alone.
- Then the engine tool-loop behind a flag (e.g. `CKAN_PERSONA_TOOLS=1`); default off keeps the current
  proven path. Flip on after the R1 probe passes.

## Open questions

1. **O1 — executor default:** in-process `ToolRegistry.invoke` (recommended) vs. force MCP-wire
   internally. (Affects latency + complexity; the abstraction supports both.)
2. **O2 — which personas get which tools** by default (author: file_* + ckan_package_search;
   evaluators: read-only or none?).
3. **O3 — MCP transport:** stdio only for now, or also a local HTTP endpoint?
4. **O4 — R1 outcome:** does the tejas endpoint support reliable tool calling? Drives whether the
   engine rewrite proceeds or tools stay MCP/evaluator-only.

## Decisions log

- 2026-06-26: forks resolved — internal MCP consumer; tool-calling personas; read+dry-run tools only
  (writes stay gated); YAML-catalog + handler tool format.

## User feedback / decisions

- 2026-06-26: User requested CKAN parts as tools via an MCP server, and the file extractors outlined
  as tools (parallel to personas) so tools are easy to add. Forks answered as above.
- _Awaiting review of this spec (and the R1 capability probe) before implementation._
