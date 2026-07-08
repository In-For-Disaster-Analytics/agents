# File/Zip Upload, Agent Triage, and a Light Chat UI

## Status

Implementing

_Increment 8a DONE: upload endpoint + safe zip extraction + head inventory.
Increment 8b DONE: GDAL tools + head-inventory→tool-calling-author triage + reviewed-files summary.
Increment 9 (code complete, needs `npm install`/manual run to verify): React/Vite chat UI in `ui/`._

## Objective

Let a user upload files (including a `.zip`) through a dedicated lightweight chat UI; the agent
safely extracts the archive, reviews the **heads** of every file cheaply, then lets the
tool-calling author **decide which files warrant deeper review** (cap ~5) — including spatial
inspection via existing extractors and a **GDAL CLI** tool when available — summarize what it
found, and discuss with the user before proposing CKAN metadata.

## User need

- The OpenAI-compatible chat box can't attach binary files (e.g. a `.zip`), so a purpose-built
  UI is needed that uploads files and calls the API.
- For a multi-file archive, the user doesn't want to hand-pick files; the agent should triage:
  skim everything, deep-dive the few that matter, use GDAL/spatial tools for geo data, then
  summarize and discuss.

## Decisions (user, 2026-06-29)

- **UI = a separate React/Vite frontend app** (not a static page, not deferred).
- **Deep-review tools = existing Python extractors + a GDAL CLI tool (`gdalinfo`/`ogrinfo`)
  used only when GDAL is on PATH** (graceful if absent).
- **Triage model**: auto-review file heads → LLM decides deeper review (≤5 files) / GDAL
  extraction → summarize → discuss (not manual user selection).
- Builds on prior decisions: per-request JWT auth, tool-calling author, schema-select, generic
  default schema.

## Current system summary (reuse)

- `app/files/` — extractor dispatch (`analyze_path`) + `safety.validate_readable_file` (size +
  sensitive-path guards). `archive.inspect_zip` lists members but does NOT extract.
- `app/tools/` — `ToolRegistry` (YAML catalog + handlers), read-only file/CKAN tools,
  `to_openai_tools()`; `InProcessToolExecutor`; the tool-calling author loop in `engine.py`.
- `app/auth_context.py` — per-request `Authorization: Bearer <jwt>` → CKAN.
- `routes_agent.py` — `/v1/chat/completions`, `/v1/ckan-registration/runs[/resume]`; per-request
  auth bound on the router.
- No upload endpoint, no extraction, no head-inventory, no GDAL tool, no UI yet.

## Proposed design

### Backend

**1. Upload endpoint** — `POST /v1/uploads` (multipart).
- Accepts one or more files; writes them under `settings.upload_root/<upload_id>/` (uuid dir).
- If a file is a `.zip`, **safely extract** it into the upload dir (see safety below).
- Returns `{upload_id, dir, files: [{name, rel_path, size, kind}], warnings}`.
- The chat/runs call then references `upload_dir = <that dir>` (existing evidence path), so the
  rest of the pipeline is unchanged.

**2. Safe zip extraction** — `app/files/archive_extract.py` (new), reused by the endpoint.
- Reject before/while extracting: total uncompressed size > cap (zip-bomb), member count > cap,
  any member with an absolute path or `..` (path-traversal), per-file size > cap, symlinks.
- Configurable caps in `Settings` (`max_zip_uncompressed_bytes`, `max_zip_members`, `max_file_bytes`).
- Returns the list of extracted file paths (confined to the upload dir).

**3. Head inventory** — cheap, bounded preview of every extracted file.
- `analyze.build_head_inventory(paths)` → for each file: name, rel_path, size, detected kind,
  and a *shallow* head (first ~1–2 KB / first rows / notebook headings) — NOT a full deep parse.
- This is the evidence the author triages from; full extractors run only on the files it chooses.

**4. Tool-calling triage (extends the existing author loop).**
- The persona/author evidence includes the head inventory + the extracted file **paths**.
- The author is given the file tools (existing `file_*` + new GDAL tools) and instructed to:
  call deeper tools on at most ~5 files that look most informative, prefer GDAL/spatial tools for
  geo files, then write metadata + a short summary of what it found. The `max_tool_calls` cap
  bounds cost; the "≤5 files" is a prompt instruction reinforced by the cap.
- A new graph step **summarize/discuss**: before (or as part of) `propose`, present a short
  human-readable summary of what the agent reviewed and found, then continue clarify→propose.

**5. New GDAL tools** — `app/tools/handlers/gdal.py` + catalog entries.
- `gdal_info(path)` → runs `gdalinfo` (raster: CRS, bounds, bands, size) via `subprocess` (args
  list, no shell), only if `gdalinfo` is on PATH; returns `{dependency_missing: "gdal"}` otherwise.
- `ogr_info(path)` → runs `ogrinfo -so -al` (vector layers/CRS/extent) similarly.
- Read-only; path-confined to the upload dir; bounded output. Added to the author tool allow-list.

### Frontend — `ui/` (new React + Vite app)

- A minimal chat window: message list, text input, a file/zip upload control, and a "thread"
  concept mapped to the API's `session_id`/thread.
- Flow: upload via `POST /v1/uploads` → get `upload_dir` → send a chat turn to
  `POST /v1/chat/completions` (or `/runs`) with that `upload_dir` in metadata. Subsequent turns
  reuse the thread; clarification/summary turns render inline.
- Auth: a field for the **CKAN JWT**, sent as `Authorization: Bearer <jwt>` on every request
  (the per-request auth we built). Base URL configurable.
- Dev: `ui/` with its own `package.json`/Vite dev server (proxy to `:8787`); production build
  output can be served statically (by FastAPI or any static host) later.

## Files likely affected

| Path | Change |
|---|---|
| `app/api/routes_uploads.py` (new) | `POST /v1/uploads` multipart endpoint |
| `app/files/archive_extract.py` (new) | safe zip extraction (caps + traversal guards) |
| `app/files/analyze.py` | `build_head_inventory`; head-vs-deep modes |
| `app/tools/handlers/gdal.py` + `catalog/gdal.yaml` (new) | `gdal_info` / `ogr_info` tools |
| `app/agents/ckan_registration/persona_nodes.py` | feed head inventory + extracted paths to the author; summarize/discuss step; widen author tool allow-list |
| `app/personas/domain_expert.md` | triage instructions (skim heads, deep-review ≤5, prefer GDAL for geo, summarize) |
| `app/settings.py` | upload/zip caps; uploads dir |
| `app/main.py` | include uploads router; (optional) serve built UI |
| `ui/` (new) | React/Vite chat app |
| tests | extraction safety, head inventory, GDAL tool (mocked subprocess), upload endpoint |

## API/schema changes

- New `POST /v1/uploads` (multipart) → `{upload_id, dir, files[], warnings}`.
- New `requires_action`/response surface for the triage **summary** (or fold into the proposal).
- New tools in the catalog (`gdal_info`, `ogr_info`).
- No change to the existing chat/runs contract (uploads reference `upload_dir`).

## Data flow

1. UI uploads files → `/v1/uploads` → safe-extract zip → `upload_dir` + file list.
2. UI sends a chat turn with `upload_dir` → schema-select → persona.
3. Persona builds head inventory of all files → tool-calling author deep-reviews ≤5 (file/GDAL
   tools) → writes metadata + summary.
4. Summarize/discuss → clarify (one question at a time) → propose → (gated) dry-run/REGISTER.

## Risks and tradeoffs

- **Zip-bomb / path-traversal / symlink** (highest): mitigated by pre-extraction caps + member
  path validation + symlink rejection in `archive_extract`. Security review before merge.
- **GDAL availability/variance**: CLI may be absent or version-variant; tool degrades gracefully
  and never shells out a user string (args list only). Output bounded.
- **Cost/latency**: triage adds tool calls; bounded by `max_tool_calls` + the ≤5 prompt rule +
  head-only review of the rest.
- **Upload abuse / disk**: size/count caps; uploads under a dedicated dir; (future) TTL cleanup.
- **Two codebases (UI + API)**: the React app adds a toolchain; kept isolated under `ui/`.
- **Scope**: this is the largest increment yet; split backend (8) from UI (9) so each ships green.

## Alternatives considered

- **Manual file selection** (earlier plan) — rejected by the user in favor of agent triage.
- **Static HTML/JS UI** — rejected in favor of React/Vite.
- **GDAL Python bindings** instead of CLI — heavier/fragile install; CLI subprocess chosen.

## Test plan

- `archive_extract`: rejects oversized/too-many-members/`..`/absolute/symlink; extracts a benign
  zip; confines paths.
- `build_head_inventory`: bounded heads for csv/notebook/pdf/text/geo.
- Upload endpoint: multipart save + zip auto-extract + response shape (FastAPI TestClient).
- GDAL tools: mocked `subprocess` (present → parsed output; absent → dependency_missing).
- Triage: with a fake tool-calling chat, author deep-reviews ≤N and summarizes; cap respected.
- Keep the full suite green (currently 87; only the unrelated ckan_guardrails artifacts fail).
- UI: component/smoke tests (Vitest) for upload + send; manual run against the dev server.

## Documentation plan

- README: the upload endpoint, the UI (`ui/` dev + build), GDAL optional dependency, the triage flow.

## Rollout / rollback plan

- Backend behind no new flag (additive endpoints/tools); the triage behavior rides the existing
  `CKAN_PERSONA_TOOLS` flag. UI is a separate app — not wired into prod serving until ready.
- Rollback: remove the uploads router / UI; the chat+runs path is unchanged.

## Open questions

1. **Summarize/discuss placement**: a distinct turn before `propose`, or a "what I reviewed"
   section appended to the proposal?
2. **Upload retention**: TTL/cleanup of `upload_root/<id>` dirs (and the file-upload-to-CKAN gap —
   should extracted files also become CKAN resources on REGISTER?).
3. **UI auth/secrets**: JWT entered in the UI is sent per request (not stored) — confirm no local
   persistence in the browser beyond the session field.
4. **Deep-review cap**: hard-cap at 5 files, or make it a setting?

## Decisions log

- 2026-06-29: UI = React/Vite separate app; deep-review = existing extractors + GDAL CLI when
  available; agent-triage (heads → ≤5 deep → summarize → discuss) over manual selection.

## User feedback / decisions

- 2026-06-29: User redirected from manual zip file-selection to (a) a light custom chat UI that
  uploads and curls the API, and (b) agent-driven triage of file heads with LLM-chosen deep review
  (≤5) using GDAL/spatial tools, then summarize and discuss.
- _Awaiting review of this spec before implementation._
