# CKAN Registration System — As-Built Design Spec

**Status:** In Review

---

## Objective

Document the existing `ckan-registration` system as it actually runs today. This is a retroactive (as-built) spec derived from direct code inspection. Every claim here was verified against the source. Nothing is invented or speculative.

---

## User Need

TACC researchers need a repeatable, LLM-assisted workflow to register scientific datasets — principally MODFLOW groundwater model file collections and Jupyter notebook outputs — as packages in the CKAN data catalog at `ckan.tacc.utexas.edu`.

Individual TACC researchers run the Jupyter notebooks interactively and occasionally, registering scientific datasets they have produced or curated. The notebooks and `ckan_agent.py` (backed by `utils.py`) are the complete system; there is no HTTP service or external orchestrator.

**File-collection / GAM registration:** `ckan_registration.ipynb` drives bulk registration of TWDB Groundwater Availability Model (GAM) file collections. It reads file bytes directly from the TACC/Corral/Lonestar6 filesystem using absolute paths from `twdb_gam_modflow_locations_with_bbox_strings.json` — a 23-model manifest with paths rooted at `/corral-repl/tacc/...` and WGS84 bounding boxes. This notebook can only run on a host that has access to those filesystem paths; it is not portable to a laptop without Corral/LS6 access.

**Notebook-to-CKAN:** `notebook_ckan_registration.ipynb` registers a Jupyter notebook itself (its cells and outputs) as a CKAN dataset, supporting reproducibility and course material archiving.

The workflow must:

1. Accept either a local file directory or a Jupyter notebook as the primary input.
2. Propose CKAN-ready metadata (title, description, tags, spatial/temporal coverage, etc.) automatically using an LLM, with a safe fallback when the LLM is unavailable.
3. Let the user review and revise the proposal before any writes occur.
4. Guard destructive CKAN writes behind explicit approval tokens.
5. Run from a TACC system (notebook or CLI) with access to the relevant local filesystem paths.

---

## Current Code / System Summary

The system is a self-contained directory (`ckan-registration/`) that does not import from the broader repo (see `README.md`: "standalone guarantee — does not import `semantic_bridge`"). All CKAN, auth, and LLM helpers live locally in `utils.py`.

### Three execution surfaces

All three surfaces share the same logical workflow: collect inputs → propose metadata via LLM → dry-run diff → apply.

#### 1. `ckan_registration.ipynb`

Interactive notebook for registering a **local file collection** (e.g., a MODFLOW GAM model directory).

- Configured via cell variables (or `.env`): resource directory path, source metadata URL, CKAN dataset name, auth credentials.
- Calls `utils.py` helper functions directly.
- Walks a directory (`list_resource_files`), builds a resource plan (`build_resource_plan`), scrapes the source URL (`fetch_source_metadata`), and proposes metadata (`propose_ckan_dataset_metadata_with_llm`).
- The companion manifest `twdb_gam_modflow_locations_with_bbox_strings.json` (23 TWDB GAM models, absolute paths on Corral/LS6) is used to drive bulk registrations of the GAM collection; file bytes are read directly from the local filesystem and uploaded to CKAN.
- Optionally enriches each resource with MINT standard variable annotations (aspirational — see MINT section).
- Dry-run guarded by an `APPLY_CHANGES = True` cell variable; stale resource deletion guarded by a separate flag.
- Default target: Yegua-Jackson Aquifer GAM dataset on TWDB.

#### 2. `notebook_ckan_registration.ipynb`

Interactive notebook for registering a **Jupyter notebook itself** as a CKAN dataset.

- Reads a target notebook from `NOTEBOOK_PATH`.
- Two-pass LLM analysis: pass 1 extracts CKAN-useful context from each cell/output chunk; pass 2 proposes package metadata.
- Normalizes spatial coverage: converts bbox strings to closed GeoJSON `Polygon` rings.
- Pulls temporal coverage only from notebook evidence (variable values, dataframe filters, displayed output dates).
- Prompts dataset titles to include location, measured quantity, units, and method/source.
- Has a hardcoded blocklist: the known TWDB GAM URL (`twdb.texas.gov/groundwater/models/gam/ygjk`) is blocked from appearing as the CKAN `url` field, even if it leaks in from `.env`, previous CKAN metadata, or LLM output.
- Can optionally upload the source notebook as a CKAN resource.
- Default target: `../DSO-Institute-2026/Day-3/Morning/3_folium_mapping.ipynb`.

#### 3. `ckan_agent.py` (Python CLI agent)

Python implementation of a **five-command surface**, calling `utils.py` directly.

| Command | Writes to CKAN? | Purpose |
|---|---|---|
| `analyze` | No | Build resource plan and propose metadata; save session state. |
| `revise` | No | Apply edits to saved state. |
| `dry-run` | No | Diff proposed vs. existing CKAN package. |
| `apply` | Yes | Create/patch package + upload resources. |
| `show` | No | Return saved session state for debugging. |

- Reads input from a file path, stdin, or base64-encoded `--input-b64` flag.
- Supports a `--secret-env-file` flag: loads a second `.env` file with `override=True`, then attempts to delete it after loading.
- Implements `apply_secret_headers()` that writes header-derived credentials into `os.environ`.
- **Note:** The module docstring still says "n8n-facing CKAN registration worker." This is stale wording from before the n8n integration was removed. The file is the CLI surface and does not depend on any deleted files. The docstring should be updated (see Documentation Plan).
- **Credential isolation:** `apply_secret_headers()` writes credentials into `os.environ` but there is no finally-block that restores them after a command completes. Safe for CLI use (each invocation is a fresh OS process), but would leak credentials across calls if `ckan_agent.py` were ever used in a persistent Python process (see Risks #2).

**Session state:** Both `ckan_agent.py` and the notebooks that use it persist state as JSON files under `/tmp/ckan-registration-agent/<session_id>.json` (overridable via `CKAN_AGENT_STATE_DIR`). Credentials are not saved to state; sensitive keys are redacted from trace events.

### `utils.py` — shared helper library

~1,965 lines. Key functional groups:

- **LLM:** `_chat_completion_content` — calls either the `openai` SDK or an HTTP fallback to `$OPENAI_BASE_URL/v1/chat/completions`. Model default: `Meta-Llama-3.3-70B-Instruct` at `ai.tejas.tacc.utexas.edu`.
- **Text/web:** `fetch_source_metadata`, `html_to_text`, `clean_text`, `slugify`, `sanitize_tag`, `dedupe_tags`.
- **File walking:** `list_resource_files`, `build_resource_plan` (notebook-path version), `summarize_extensions`.
- **Metadata proposal:** `propose_ckan_dataset_metadata_with_llm` — sends a detailed system prompt that instructs the LLM not to invent dates, authors, or spatial coverage; explicit `null` fallback rules for unknown fields.
- **CKAN auth:** `get_tapis_token` (Tapis password→bearer), `build_ckan_auth_header` (two modes: `api_token` or `tapis_password`).
- **CKAN read/write:** `fetch_ckan_dataset`, `fetch_existing_dataset_or_none`, `create_or_update_ckan_dataset` (tries `package_show`; if dataset missing calls `package_create`; if present calls `package_patch` + optional `package_owner_org_update`), `upsert_resources` (per-file `resource_create` or `resource_update`), `remove_stale_resources` (`resource_delete`).
- **Diff/review:** `compare_dataset_metadata`, `render_changes_table_markdown`.
- **MINT (read-only, ~800 lines):** hierarchy traversal (models → versions → configurations → setups → dataset specifications), standard variable lookup, SVO name inference from file extensions, SVO scoring and matching per resource file. All MINT calls are best-effort (`try/except` around every network call).

---

## As-Built Design

### Core workflow (all surfaces)

```
[Input]
  local files / upload_dir / source_url / message text
       │
       ▼
[collect_file_records / list_resource_files]
  walk directories, skip hidden files, deduplicate
       │
       ▼
[build_resource_plan]
  per-file: relative path, SHA-256, size, MIME type,
  text preview (up to 1 800 chars), inferred tags
       │
       ▼
[propose_metadata]
  1. compute fallback metadata (title from message/filenames, slugified name)
  2. if OPENAI_API_KEY set and resource plan non-empty:
       fetch source URL → scrape title + 6 000-char excerpt
       call LLM with detailed system prompt
       parse JSON response; merge non-null LLM values over fallback
  3. apply any caller-supplied dataset overrides (field-level, allowlisted)
       │
       ▼
[save_state  →  /tmp/ckan-registration-agent/<session_id>.json]
       │
       ▼
[dry_run]
  call package_show on target dataset name
  compare_dataset_metadata → field-by-field diff table
  resource_delta → create / update / delete_candidates sets
       │
       ▼
[apply]  ← requires approval == "REGISTER"
  create_or_update_ckan_dataset (package_create or package_patch)
  upsert_resources (resource_create or resource_update per file)
  optional: remove_stale_resources ← requires delete_approval == "DELETE_STALE_RESOURCES"
```

### MINT enrichment (aspirational — not yet used in practice)

The codebase contains ~800 lines of MINT integration (`utils.py`). This capability was built for future use; it is not regularly exercised in practice today, and no MINT write path is intended. It is read-only and best-effort.

When enabled in `ckan_registration.ipynb` (opt-in via notebook cell variables):

1. Infer model label from file extensions/names (heuristic) or LLM.
2. Look up standard variable names via `get_mint_standard_variables_for_model`:
   - Try `/custom/models/standard_variable` endpoint.
   - Try model configuration hierarchy (models → softwareversions → modelconfigurations → modelconfigurationsetups → datasetspecifications).
   - Load from local `model_yamls/*.yaml` if present.
   - Use caller-supplied fallback names.
3. Score each resource file against each standard variable name (extension matching, filename token matching).
4. Annotate resource plan items with `mint_standard_variables` and `mint_model_label` fields.
5. These fields are written as custom resource fields on `resource_create`/`resource_update` via `upsert_resources`.

Because this path is not regularly used, its correctness in practice is unverified. All MINT calls are wrapped in `try/except`; failures are silent. No MINT write path (registering CKAN datasets back into MINT) is planned.

---

## Files / Components

| File | Role |
|---|---|
| `utils.py` | Core Python helper library (CKAN, LLM, MINT, auth, file, text). ~1 965 lines. |
| `ckan_agent.py` | Python CLI agent. Five-command surface (`analyze`, `revise`, `dry-run`, `apply`, `show`). ~1 522 lines. |
| `ckan_registration.ipynb` | Interactive notebook: local file-collection → CKAN registration (GAM/MODFLOW use case). |
| `notebook_ckan_registration.ipynb` | Interactive notebook: Jupyter notebook → CKAN registration. |
| `twdb_gam_modflow_locations_with_bbox_strings.json` | 23-model GAM manifest: package IDs, Corral/LS6 absolute directory paths, TWDB source URLs, WGS84 bboxes. Used by `ckan_registration.ipynb` to drive bulk GAM registration. |
| `.env.sample` | Environment variable template for local use. |
| `data/` | Sample MODFLOW model files (Yegua-Jackson) for local testing without Corral access. |

---

## External APIs and Schemas Used

### CKAN REST API (`ckan.tacc.utexas.edu`)

| Endpoint | Direction | Purpose |
|---|---|---|
| `GET /api/3/action/package_show?id=<name>` | Read | Fetch existing dataset for diff or update. |
| `POST /api/3/action/package_create` | Write | Create new dataset. |
| `POST /api/3/action/package_patch` | Write | Update existing dataset fields. |
| `POST /api/3/action/package_owner_org_update` | Write | Transfer dataset to a different org. |
| `POST /api/3/action/resource_create` | Write | Upload new resource file. |
| `POST /api/3/action/resource_update` | Write | Replace/update existing resource file. |
| `POST /api/3/action/resource_delete` | Write | Delete stale resource. |

Auth header: plain API token string (for `api_token` mode) or `Bearer <tapis_jwt>` (for `tapis_password` mode).

CKAN dataset fields written: `name`, `title`, `notes`, `url`, `owner_org`, `private`, `tags`, `author`, `author_email`, `maintainer`, `maintainer_email`, `license_id`, `version`, `type`, `isopen`, `spatial`, `temporal_coverage_start`, `temporal_coverage_end`.

### Tapis OAuth2 (`portals.tapis.io`)

- `POST /v3/oauth2/tokens` with `grant_type=password`.
- Called only when `CKAN_AUTH_MODE=tapis_password`.
- Returns a JWT used as `Bearer` token for CKAN.

### OpenAI-compatible LLM (`ai.tejas.tacc.utexas.edu`)

- `POST /v1/chat/completions` with model `Meta-Llama-3.3-70B-Instruct` (default).
- Called via the `openai` Python SDK when installed; falls back to direct `requests.post` in `utils.py` when the SDK is absent.
- System prompt explicitly instructs the model not to invent dates, authors, emails, or spatial/temporal coverage. Unknown fields must be returned as `null`.

### MINT API (`api.models.mint.tacc.utexas.edu/v2.0.0`) — read-only

| Endpoint | Purpose |
|---|---|
| `GET /models?label=<label>` | List models by label. |
| `GET /softwares?label=<label>` | List software entries by label. |
| `GET /softwareversions/<id>` | Fetch model version detail. |
| `GET /modelconfigurations?label=<label>` | List model configurations. |
| `GET /custom/modelconfigurations/<id>` | Fetch configuration (preferred). |
| `GET /modelconfigurationsetups/<id>` | Fetch setup detail. |
| `GET /custom/modelconfigurationsetups/<id>` | Fetch setup (preferred). |
| `GET /datasetspecifications/<id>` | Fetch dataset spec for a presentation. |
| `GET /custom/models/standard_variable?label=<label>` | Direct standard variable lookup. |
| `GET /standardvariables?label=<label>` | Resolve SVO name to canonical ID. |

All MINT calls are wrapped in `try/except` and silently skipped on failure (best-effort enrichment).

---

## Data Flow

### File-collection path (`ckan_registration.ipynb` or `ckan_agent.py`)

For GAM registrations: resource files reside on the Corral/Lonestar6 filesystem (absolute paths from `twdb_gam_modflow_locations_with_bbox_strings.json`). File bytes are read and uploaded directly; the process must run on a host with those paths mounted.

```
Local filesystem (or Corral/LS6 for GAM registrations)
  └─ resource files (up to 5 000 per session)
       │  sha256, size, MIME, text preview (1 800 chars max)
       ▼
resource_plan  (list of dicts; stored in session state JSON)
       │
       ├─ [aspirational / not yet used] MINT API  →  standard_variable_ids / names per resource file
       │
       ├─ source URL  →  HTTP GET  →  title + 6 000-char page excerpt
       │
       └─ LLM  →  proposed metadata dict
                      │
                      ▼
               desired_dataset_payload  (stored in session state JSON)
                      │
              [dry_run]  →  CKAN package_show
                      │           │
                      │    compare_dataset_metadata
                      │    resource_delta
                      │
              [apply]  →  CKAN package_create / package_patch
                       →  CKAN resource_create / resource_update (multipart upload)
                       →  [optional] CKAN resource_delete
```

### Session state

State files are JSON objects at `/tmp/ckan-registration-agent/<session_id>.json`. Schema version `1`. Key fields:

```json
{
  "schema_version": 1,
  "session_id": "...",
  "status": "analyzed | dry_run | revised | applied",
  "message": "...",
  "source_urls": ["..."],
  "existing_ckan_entry": "dataset-name-or-id",
  "ckan": { "url": "...", "owner_org": "...", "upload_resources": true, "remove_stale_resources": false },
  "llm_dataset": { ... },
  "desired_dataset_payload": { ... },
  "resource_plan": [ { "resource_name": "...", "local_path": "...", ... } ],
  "warnings": [ "..." ],
  "trace": [ { "at": "...", "step": "..." } ]
}
```

Credentials are never written to state files. Sensitive keys are redacted from trace events (`SENSITIVE_TRACE_KEYS` set in `ckan_agent.py`).

---

## Risks and Tradeoffs

### High risk

1. **`apply` is irreversible.** There is no rollback mechanism in the code. `package_patch` overwrites metadata fields; `resource_delete` permanently removes files from CKAN storage. The only mitigation is the double-approval token requirement (`REGISTER` for apply, `DELETE_STALE_RESOURCES` for stale removal).

2. **`ckan_agent.py` has no credential restore after `apply_secret_headers`.** `apply_secret_headers()` writes header-derived credentials into `os.environ` but there is no finally-block to restore them. Safe for current CLI use (each invocation is a fresh OS process), but would leak credentials across calls if `ckan_agent.py` were ever used in a persistent Python process (e.g., imported as a library or integrated into a long-running service). Should be addressed if usage evolves beyond single-shot CLI invocations.

3. **README references deleted files — active doc gap.** `README.md` still references `ckan_agent.mjs`, `ckan_agent_server.mjs`, and the `n8n/` directory, all of which have been deleted. It also describes the JS worker as primary and Python as the fallback, which is the opposite of the correct framing. The README must be updated before new contributors use it.

### Medium risk

4. **LLM hallucination.** The LLM is explicitly instructed not to invent facts and to return `null` for unknown fields. A heuristic fallback covers LLM failures. However, the LLM may still produce plausible-but-wrong values that pass silently (e.g., a wrong spatial extent or author name). The dry-run diff step is the intended human review gate.

5. **No test suite.** There are no automated tests in the directory (no `test_*.py`, no test runner configuration). Correctness depends entirely on manual runs and the dry-run review step.

6. **GAM registration requires Corral/LS6 filesystem access.** `ckan_registration.ipynb` reads model files directly from `/corral-repl/tacc/...` paths. Running it from a host without those paths mounted will fail silently at the file-walk step (zero resources collected) rather than raising a clear error. The bundled `data/` sample files allow smoke-testing metadata proposal without Corral access.

### Low risk

7. **MINT enrichment unverified in practice.** MINT is aspirational and not regularly used. All calls are best-effort and silently skipped on failure. If/when it is used, correctness of the SVO matching heuristics has not been validated against real CKAN registrations.

8. **Source URL scraping.** `fetch_source_metadata` is a simple HTML-strip approach; it can produce noisy excerpts. The LLM prompt includes explicit rules to discard cookie banners, navigation, and JavaScript warnings.

---

## Alternatives Considered

No formal alternatives document exists. From README and code comments, these design choices are evident:

- **n8n integration (tried and removed).** A Node.js HTTP worker (`ckan_agent.mjs` + `ckan_agent_server.mjs`) and n8n workflow were built to support chat-driven CKAN registration. This was removed after being judged a failed experiment; the system is now notebook/Python-only. See Decisions.
- **Python SDK vs. raw HTTP for LLM.** `utils.py` tries the `openai` SDK first and falls back to `requests.post` if the SDK is absent. This allows the library to work in environments where the SDK is not installed.
- **Standalone vs. shared library.** The directory deliberately does not import from `semantic_bridge` or any other sibling module. The README documents this as the "standalone guarantee." This increases code duplication (helpers copied locally) but makes the directory independently deployable.

---

## Test Plan

There is no automated test suite. Manual smoke-test procedure (using the bundled `data/` sample files, which do not require Corral/LS6 access):

```bash
python ckan_agent.py analyze --input <(echo '{"session_id":"test","upload_dir":"data/"}') --no-llm
python ckan_agent.py dry-run --input <(echo '{"session_id":"test"}')
```

**Recommended tests to add** (not currently present):

1. Unit tests for `utils.py`: `compare_dataset_metadata`, `dedupe_tags`, `slugify`, `_parse_llm_json`, `infer_standard_variable_names_from_resource_files`, `match_standard_variables_for_resource`.
2. Unit tests for `ckan_agent.py`: `analyze_preflight_issue`, `fallback_metadata`, `resource_delta`, approval token guards.
3. Integration smoke test: `python ckan_agent.py analyze --no-llm` against the bundled `data/` directory must exit 0 and produce valid JSON with `"ok": true`.
4. Test that `apply` rejects requests without `"approval": "REGISTER"`.
5. Test that `remove_stale_resources` rejects requests without `"delete_approval": "DELETE_STALE_RESOURCES"`.

---

## Documentation Plan

Existing documentation:

- `README.md` — workflow overview, file inventory, CLI quick-start, default target values. **Currently stale** (references deleted files).
- `.env.sample` — environment variable reference with defaults.

Gaps / active work needed:

- **`README.md` must be rewritten.** It references `ckan_agent.mjs`, `ckan_agent_server.mjs`, and the `n8n/` directory (all deleted). It also calls the JS worker primary and Python the fallback (inverted). The file inventory, usage instructions, and architecture overview all need updating to reflect the current three-surface, Python-only system.
- **`ckan_agent.py` module docstring** says "n8n-facing CKAN registration worker." This is stale; the docstring should describe it as the Python CLI agent.
- No architecture diagram.
- `ckan_agent.py`'s credential isolation behavior (`apply_secret_headers` with no env restore) is not documented anywhere; a developer note or code comment would reduce future confusion.
- MINT is not characterized as aspirational in any doc; `README.md` describes it as a current feature without qualification.
- The accepted recovery procedure for a bad `apply` (re-run with corrected inputs) should be documented explicitly in `README.md`.

---

## Rollout / Rollback Plan

**Rollout (current state):**

- Notebooks and `ckan_agent.py` run on TACC systems (Lonestar6 or similar) by individual researchers with a populated `.env`.
- For GAM registrations, the process must run on a host with Corral filesystem access (`/corral-repl/tacc/...`). The bundled `data/` sample files allow local testing of the metadata pipeline without Corral access.
- There is no deployed service or scheduled process; all invocations are researcher-initiated.

**Recovery from a bad CKAN apply (accepted procedure):**

There is no automated rollback and none is planned. The accepted recovery path is:

1. Correct the input data, metadata overrides, or `.env` configuration.
2. Re-run `apply`. Because `create_or_update_ckan_dataset` calls `package_patch` when the dataset already exists, a re-run with corrected inputs will overwrite the bad metadata fields.
3. For newly created datasets that should be removed entirely: call `package_delete` via the CKAN API or admin UI.
4. For stale resources deleted by `remove_stale_resources`: these are permanently gone from CKAN storage. Re-upload from the original local source if needed. This is why stale deletion requires the explicit `delete_approval: "DELETE_STALE_RESOURCES"` token.

**Optional future enhancement:** A `--backup` or `--snapshot` flag that saves the existing CKAN package JSON before patching would reduce the risk of unrecoverable metadata loss and lower the cost of re-running with corrections. This is not currently implemented.

---

## Open Questions

The following questions were resolved by the maintainer on 2026-06-25 and moved to Decisions:
- ~~Q1: Deployment context~~ → Resolved: notebook-first internal tool; n8n is secondary.
- ~~Q3: Python vs. JS primary surface~~ → Resolved: Python is primary; JS is n8n-only; README is wrong.
- ~~Q5: MINT integration status~~ → Resolved: aspirational, not yet used in practice, no write path intended.
- ~~Q8: Rollback story~~ → Resolved: re-run with corrected inputs is the accepted procedure.

The following questions remain open:

1. **Original motivation for the notebook-to-CKAN path.** The `notebook_ckan_registration.ipynb` default target is a file from `DSO-Institute-2026`. Was this built for a specific course or reproducibility initiative? The answer affects whether the TWDB URL blocklist is a one-off guard or a pattern to generalize.

2. **Meaning of the "standalone guarantee" in practice.** Is the no-`semantic_bridge`-import constraint a deliberate deployment requirement (the directory must be runnable without the full repo), a testing constraint, or a historical artifact?

3. **Known edge cases or bugs to document.** Are there known edge cases in the MINT hierarchy traversal that produce incorrect SVO annotations? Are there other known limitations that should be explicitly documented for future maintainers?

---

## Decisions

| Date | Decision | Rationale |
|---|---|---|
| (pre-spec) | Standalone guarantee: no `semantic_bridge` import. | Enables independent deployment without the full repo present. |
| (pre-spec) | Double-approval tokens for destructive operations. | `REGISTER` for apply; `DELETE_STALE_RESOURCES` for stale removal. Reduces accidental data loss. |
| (pre-spec) | MINT integration is best-effort / read-only. | MINT API availability is not guaranteed; silently skip on failure rather than blocking registration. |
| (pre-spec) | LLM prompt instructs model to return `null` for unknown fields, not to invent facts. | Guards against hallucinated authorship, temporal coverage, or spatial extent being silently written to CKAN. |
| 2026-06-25 | **Primary surface is notebooks + Python (`utils.py` / `ckan_agent.py`).** | Confirmed by maintainer. README's opposite framing was incorrect and is now an active doc gap. |
| 2026-06-25 | **MINT integration is aspirational — not yet used in regular practice.** No MINT write path is planned. | Confirmed by maintainer. Built for future use; not regularly exercised today. |
| 2026-06-25 | **Accepted recovery for a bad `apply` is to re-run with corrected inputs.** No automated rollback is planned. | Confirmed by maintainer. `package_patch` overwrites fields; a corrective re-run is sufficient. Deleted resources require manual re-upload. |
| 2026-06-25 | **n8n integration and the Node agents (`ckan_agent.mjs`, `ckan_agent_server.mjs`) were deleted.** n8n was a failed experiment. The system is now notebook/Python-only. | Files removed: `ckan_agent.mjs`, `ckan_agent_server.mjs`, entire `n8n/` directory. `ckan_agent.py` is retained as the CLI surface and does not depend on the deleted files. `README.md` must be updated to remove all n8n/.mjs references. |

---

## User Feedback / Decisions

**2026-06-25 — Maintainer answered four open questions:**

1. **Deployment context (was Q1):** Notebook-first internal tool. TACC researchers run the notebooks occasionally. The n8n/HTTP agent is a secondary surface, not the production centerpiece. Incorporated into User Need, Rollout, and surface priority throughout.

2. **Python agent future (was Q3):** Python is primary. `utils.py` + `ckan_agent.py` + the notebooks are the maintained surface. `ckan_agent.mjs` + `ckan_agent_server.mjs` exist only for the n8n integration. The README's opposite framing is a doc gap that should be corrected. The missing `restoreEnv` in `ckan_agent.py` is a more relevant risk given Python is primary (kept at High risk). The JS LLM prompt parity gap is lower priority (downgraded to Low risk).

3. **MINT integration (was Q5):** Aspirational. The ~800 lines of MINT code were built for future use; MINT enrichment is not regularly used in practice today. No MINT write path is intended. Characterized as aspirational throughout the spec.

4. **Rollback story (was Q8):** Re-run with corrected inputs is the accepted recovery procedure. `package_patch` overwrites fields, so a corrective re-run is sufficient for metadata errors. Deleted stale resources require manual re-upload. No automated rollback is planned. Pre-apply snapshot noted as a future enhancement only.

**2026-06-25 — n8n integration deleted:**

The n8n integration (Node.js agents + n8n workflow) was confirmed a failed experiment and removed. Deleted files: `ckan_agent.mjs`, `ckan_agent_server.mjs`, `n8n/` directory. `ckan_agent.py` is retained as the CLI surface and does not depend on the deleted files, though its module docstring still contains stale "n8n-facing" wording. The system is now three-surface, notebook/Python-only. Spec updated throughout to remove all n8n/Node content; `README.md` update is now an active doc gap.
