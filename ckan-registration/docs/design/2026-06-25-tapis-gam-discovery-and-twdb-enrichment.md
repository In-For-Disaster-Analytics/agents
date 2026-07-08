# Lonestar6 GAM Discovery, TWDB Enrichment, and Persona-Evaluated SUBSIDE Metadata

> **Note on filename:** The filename (`2026-06-25-tapis-gam-discovery-and-twdb-enrichment.md`) retains "tapis" for cross-link stability with the retroactive spec. The approach no longer uses Tapis Files for data access; Tapis is used only for CKAN authentication.

**Status:** Implementing

**Current system reference:** `docs/design/2026-06-25-ckan-registration-system.md`

> **Implementation note (2026-06-25):** Capabilities A, B, D implemented as modules (`discovery.py`, `aquifer.py`, `pdf_extract.py`, `twdb_enrich.py`, `subside_mapping.py`, `persona_loop.py`) with orchestration in `orchestrate.py`; 117 unit tests pass. **Deviation:** Capability C's schema change is captured as an in-repo proposed artifact at `schema/subside_dataset.proposed.yaml` (NOT applied to the shared `ckanext-dso_scheming` repo, per user direction to keep all edits within `ckan-registration`); deploy + migration remain gated. `quality_control_level` choices resolved from the SUBSIDE spreadsheet. Notebook (`ckan_registration.ipynb`) wired to the pipeline (dry-run by default). Project reorganized into a `src/gam_registration/` package (`notebooks/`, `tests/`, `data/`, `fixtures/`, `schema/`, `archive/`, `pyproject.toml`; `pytest` via `pythonpath=src`; CLI `ckan-agent`). Remaining: live Lonestar6 + CKAN dry-run/apply (gated), and the `dso_scheming` schema deploy + migration (gated).

---

## Objective

Extend the GAM registration workflow with four capabilities: (A) automate discovery and manifest generation on Lonestar6 using a local recursive directory walk with best-effort spatial derivation, (B) enrich CKAN metadata with TWDB report content via a map-reduce LLM strategy, (C) extend the live `subside_dataset` ckanext-scheming schema to the full 28-field SUBSIDE specification, and (D) replace the single metadata-proposal call with a three-persona iterative authoring and evaluation loop that serves as a **prototype and evaluation harness for a future autonomous metadata agent**.

The tool runs on Lonestar6 (or any host with the GAM files locally mounted). Tapis is retained for CKAN authentication only; Tapis Files API is not used.

The overall feature is explicitly treated as a test bed: Capability D's audit trail and convergence metrics are first-class outputs alongside the registered CKAN datasets.

This is a proposed feature spec. Do not implement until the spec is Approved.

---

## User Need

The existing `ckan_registration.ipynb` workflow requires running on a TACC system (LS6 or equivalent) that has the Corral filesystem mounted at `/corral-repl/tacc/...`. This limits who can run it and from where. Additionally, the 23-model manifest (`twdb_gam_modflow_locations_with_bbox_strings.json`) was produced by a one-time external process and must be manually maintained as the Corral collection grows. Finally, the current LLM metadata proposal uses only the TWDB landing page HTML; it cannot read the associated GAM report PDFs, which contain richer scientific context.

Users need:

1. A way to auto-discover GAM models from a local directory tree on Lonestar6 and produce or refresh the manifest, rather than maintaining it by hand.
2. Richer CKAN metadata derived from GAM report PDFs, and CKAN link resources pointing to both the TWDB landing page and the report PDF for each registered GAM.
3. GAM registrations that conform to the full SUBSIDE metadata standard — with the correct `subside_dataset` package type, all 28 documented SUBSIDE dataset-level fields present in the schema, controlled vocabularies enforced, and GAM-appropriate defaults applied (`collection_method = "Model Output"`, `categories` including `"Groundwater"`).
4. A metadata quality gate where LLM-proposed metadata is reviewed by two independent evaluator personas before being presented to the user — and an audit trail of those evaluations for assessing the prototype's effectiveness.

The system continues to run on Lonestar6 (the same run environment as the existing workflow); the goal is to automate what was previously manual, not to change the deployment model.

---

## Current Code / System Summary

The following code is directly affected. All line numbers verified against the current source.

### `utils.py`

- **`list_resource_files(resource_dir, max_files)`** (line 143): walks the local filesystem with `Path.rglob("*")`. Raises `FileNotFoundError` if the path does not exist on the running host. No network access.
- **`build_resource_plan(files, root, source_url)`** (line 158): produces the resource plan list from local `Path` objects. Each item carries `"local_path": path` (a `Path`).
- **`upsert_resources(...)`** (line 1881): opens `item["local_path"]` with `local_path.open("rb")` (line 1925) and passes the file handle as a multipart upload. Requires the path to be readable on the local filesystem.
- **`fetch_source_metadata(url)`** (line 118): fetches a URL, strips HTML, extracts `<title>`, meta description, and a 6 000-char page excerpt. Does not follow links; does not fetch PDFs.
- **`get_tapis_token(username, password, tapis_url)`** (line 1267): POSTs to `portals.tapis.io/v3/oauth2/tokens` with `grant_type=password` and returns the access token string. This JWT is used to build the CKAN `Authorization` header (via `build_ckan_auth_header`). This is the only Tapis usage in the system; the Tapis Files API is not used and will not be used in this feature.
- **`propose_ckan_dataset_metadata_with_llm(...)`** (line 1349): accepts a `source_metadata_url` parameter; internally calls `fetch_source_metadata` to get landing-page text, then builds the LLM prompt. There is no mechanism to inject PDF text alongside the landing-page excerpt.

### `ckan_registration.ipynb`

- Reads `twdb_gam_modflow_locations_with_bbox_strings.json` to get the list of 23 GAM models with their `package_folder` absolute paths and `twdb_page_url` values.
- Iterates over models; for each, calls `list_resource_files(Path(model["package_folder"]))` and `build_resource_plan(...)`, then calls the LLM metadata proposal and CKAN upsert functions.
- Requires the `package_folder` path (e.g., `/corral-repl/tacc/aci/PT2050/...`) to be readable on the running host.

### `twdb_gam_modflow_locations_with_bbox_strings.json`

Current schema per model record:
```json
{
  "package_id": "blossom-aquifer-gam",
  "package_folder": "/corral-repl/.../Blossom_Aquifer_GAM",
  "title": "Blossom Aquifer GAM",
  "twdb_page_url": "https://www.twdb.texas.gov/groundwater/models/gam/blsm/blsm.asp",
  "twdb_page_type": "gam",
  "url_verified_from_twdb": true,
  "namefile_count": 1,
  "modflow_model_directory_count": 1,
  "modflow_model_directories": [ { "relative_directory": "...", "absolute_directory": "...", "namefiles": [...] } ],
  "package_folder_name": "Blossom_Aquifer_GAM",
  "boundary_match": { ... },
  "boundary_bbox_wgs84": { "min_lon": ..., "min_lat": ..., "max_lon": ..., "max_lat": ... },
  "boundary_bbox_geojson": "{ ... }",
  "dataset_spatial": "{ ... }"
}
```

`boundary_bbox_wgs84`, `boundary_bbox_geojson`, and `dataset_spatial` were populated by an external one-time process. Recursive file discovery alone cannot reproduce these.

### `subside_dataset.yaml` (separate repo — verified 2026-06-25)

**Location:** `/Volumes/Macintosh HD - Data/Github/modflow-suite/ckan-docker/src/ckanext-dso_scheming/ckanext/dso_scheming/subside_dataset.yaml`
**Live on:** `ckan.tacc.utexas.edu` (user-confirmed; `dataset_type: subside_dataset`, `scheming_version: 2`)

Currently deployed dataset-level fields (15): `title`, `name`, `notes`, `tag_string`, `license_id`, `owner_org`, `url`, `version`, `author`, `author_email`, `maintainer`, `maintainer_email`, `temporal_coverage_start`, `temporal_coverage_end`, `mint_standard_variables`.

Currently deployed resource-level fields (15): `name`, `url`, `abstract`, `format`, `temporal_coverage_start`, `temporal_coverage_end`, `program_area`, `data_contact_email`, `caveats_usage`, `categories`, `primary_tags`, `secondary_tags`, `collection_method`, `quality_control_level`, `spatial` (`json_object` preset).

All classification/spatial fields (`program_area`, `data_contact_email`, `caveats_usage`, `categories`, `primary_tags`, `secondary_tags`, `collection_method`, `quality_control_level`, `spatial`) are currently placed at **resource level** in the deployed YAML. All these fields are free-text (no `choices`). The `spatial` field uses the `json_object` preset.

The current `ckan-registration` code creates packages with the default CKAN `dataset` type, not `subside_dataset`. This means existing GAM registrations are not using this schema.

**SubsideWiki documented target (28 dataset-level fields):** Title, Name, URL, UUID, Publishing Status, Organization, Abstract, Program Area, Data Contact Email, Caveats and Usage, Categories, Primary Tags, Secondary Tags, Bounding Coordinates, Coordinate System, Coordinate Field Format, Collection Method, Collection Method Description, Quality Control Level, Quality Assurance Description, Update Frequency, Date Range, From Date, To Date, Disclaimer, Additional Information, Related Resources, Supporting URL.

The wiki documents most classification fields at dataset level; the deployed YAML places them at resource level. This is a placement conflict that must be resolved as part of Capability C (see OQ-1).

Wiki sources: `SubsideWiki/Wiki/metadata-schema.md`, `subside-metadata-schema-spreadsheets.md`, `metadata-implementation-decisions.md`, `metadata-review-ckan-alignment.md`.

---

## Proposed Design

### Run environment

The tool runs on Lonestar6 (or any host where the Corral GAM filesystem is locally mounted). Files are read with `local_path.open("rb")` — the existing pattern in `upsert_resources`. Tapis is used exclusively for CKAN authentication (`get_tapis_token` → JWT → `Authorization` header). No Tapis Files API calls are made.

Outbound network access is required from the run host for: TWDB landing page and PDF fetches (Capability B), LLM persona calls (Capability D), and CKAN API writes. A Lonestar6 login node satisfies all these requirements. An isolated compute node without outbound internet does not.

**New `.env` variable:**
```
GAM_ROOT_DIR=/corral-repl/tacc/aci/PT2050/projects/PTDATAX-286/twdb_gam_collection
```

---

### Capability A — Local recursive discovery generates/refreshes the manifest

**Goal:** replace the one-time-generated manifest with a reproducible local-scan-based manifest generator, while keeping the manifest as a human-reviewable checkpoint before registration.

**New function in `utils.py`:**

```python
def discover_gam_models_from_local(
    root_path: Path,
    *,
    max_files: int = 50000,
) -> dict[str, Any]:
    """
    Recursively walk root_path using Path.rglob("*").
    Identify MODFLOW model directories by detecting .nam / .NAM files.
    Attempt bbox derivation from the corresponding .dis file using flopy.
    Write and return a fresh manifest dict; does not read or merge any existing manifest.
    """
    ...
```

**Model identification rule:** A directory subtree containing at least one MODFLOW `.nam` / `.NAM` namefile constitutes a model. The containing top-level subdirectory of `root_path` is the GAM package folder. (Resolved — see Decisions.)

**Produced manifest fields per model:**
```json
{
  "package_id": "<slugified top-level directory name>",
  "package_folder": "<absolute path>",
  "title": "<human-readable from directory name>",
  "modflow_model_directories": [ { "relative_directory": "...", "namefiles": [...] } ],
  "twdb_page_url": "",
  "report_url": "",
  "boundary_bbox_wgs84": "<derived from DIS file via flopy, or null if derivation fails>",
  "boundary_bbox_geojson": "<derived or null>",
  "dataset_spatial": "<derived or null>",
  "bbox_derivation_status": "ok | failed_no_dis | failed_no_crs | failed_error"
}
```

**Manifest generation strategy (simplified — no merge):** Each run produces a fresh manifest and overwrites any existing one. Manually edited values (`twdb_page_url`, `report_url`, corrected bboxes) must be re-applied after regeneration. The tool is a generator, not an authoritative store.

**Bbox derivation — three-step fallback chain:**

Step 1 is attempted first; each step falls through to the next on failure.

**Step 1 — Primary: MODFLOW DIS via flopy** (`bbox_derivation_status = "ok_from_dis"`)

After identifying each model's `.dis` file in the local directory tree, use `flopy` to read the DIS object and extract NROW, NCOL, DELR, DELC. Combine with grid origin (xoff/yoff) and CRS to compute a WGS84 bounding box.

**Texas-bounds sanity check (triggers fallback):** After computing the bbox from DIS, apply two rejection tests:
1. `xoff == 0.0 and yoff == 0.0` — flopy returns 0.0/0.0 when the origin is absent from the DIS file. This produces a grid that begins at the prime meridian/equator — clearly wrong for Texas models.
2. Computed bbox center falls outside the approximate Texas bounding box (lon: −107..−93, lat: 25..37).

If either test fails, the DIS-derived bbox is **rejected as suspicious** and the process falls through to Step 2. Status is NOT set to `ok_from_dis`; a diagnostic `bbox_derivation_status = "suspicious_dis"` is recorded and Step 2 is attempted.

**Step 2 — Fallback: aquifer-boundary lookup** (`bbox_derivation_status = "ok_from_aquifer"`)

Map the GAM `package_id` to its Texas aquifer(s) using a curated lookup table (`gam_aquifer_map.json`). Fetch the aquifer polygon(s) from the TWDB ArcGIS FeatureServer (accessed via STAC item asset URL) and compute a WGS84 bounding box from the polygon coordinates using `min/max` over all coordinate pairs (no additional library needed).

See new Capability A2 section below for the data source specifics.

**Step 3 — No spatial** (`bbox_derivation_status = "failed_no_spatial"`)

If both Step 1 and Step 2 fail, spatial fields are set to `null`. The model is flagged for manual review. Registration proceeds without spatial coverage.

**Summary of `bbox_derivation_status` values:**

| Value | Meaning |
|---|---|
| `ok_from_dis` | DIS-derived bbox passed Texas-bounds check |
| `suspicious_dis` | DIS bbox rejected (zero origin or outside Texas); aquifer fallback was tried |
| `ok_from_aquifer` | Aquifer-boundary bbox used (DIS absent, failed, or suspicious) |
| `failed_no_spatial` | Both DIS and aquifer lookup failed; null spatial |

**New environment variable (optional):** `MODFLOW_DEFAULT_EPSG` — EPSG code to assume when no CRS is declared in the model files. Applies only to Step 1 before the sanity check.

---

### Capability A2 — Aquifer-boundary bbox fallback (STAC + TWDB ArcGIS)

**Goal:** when DIS-based bbox derivation fails or produces suspicious results, compute a coarser but reliable bounding box from the aquifer's polygon extent.

**Caveat to document prominently:** The aquifer extent covers the WHOLE aquifer, which is coarser than the actual model domain. The `ok_from_aquifer` status makes this transparent. The aquifer bbox is acceptable for spatial discovery and search; it is not a precise model-domain footprint.

**Data source:**
- STAC API: `https://stacapi.pods.portals.tapis.io/api/v1`
- Collection: `subside-context`
- Items: `major-aquifers` and `minor-aquifers`
- Each STAC item's `asset["service"]["href"]` points to a TWDB ArcGIS FeatureServer returning GeoJSON of ALL aquifers in the layer (statewide). Example verified URL for minor aquifers: `https://services1.arcgis.com/7DRakJXKPEhwv0fM/ArcGIS/rest/services/Z_Statewide_gdb/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&maxAllowableOffset=0.002&geometryPrecision=5&f=geojson`. The major-aquifers item has an analogous asset. **The implementation must read the asset href from the STAC item, not hardcode the ArcGIS URL**, so the source stays single-sourced.

**Implementation approach:**
1. Fetch the STAC collection once per run (or per discovery session) to get the `major-aquifers` and `minor-aquifers` item asset URLs.
2. Fetch each statewide FeatureServer GeoJSON **once** and cache it in memory for the run (do NOT re-download for each of the 23 GAM models).
3. Look up the aquifer by name using a `where` clause filter on the ArcGIS query OR by filtering the cached GeoJSON features client-side by the aquifer-name field.

**RESOLVED 2026-06-25:** The ArcGIS attribute field for aquifer name was confirmed by inspecting `FeatureServer/0?f=json`. The minor-aquifers layer (`Minor_Aquifers`) exposes: `AQU_NAME` (string, primary aquifer name), `AQ_NAME_UL` (string, alternate), and `AQUIFER` (integer code). **Use `AQU_NAME` for the where-clause / client-side filter** (case-insensitive recommended, e.g. `UPPER(AQU_NAME) = UPPER('<name>')`). Major vs. minor is determined by which STAC item / ArcGIS layer is queried, not by a field. Confirm the major-aquifers layer uses the same field when implementing (it likely shares this schema, but verify against its own `FeatureServer/0?f=json`).

**Bbox computation:** Once the matching aquifer polygon GeoJSON feature is found, compute the bbox as:
```python
min_lon = min(coord[0] for ring in polygon["coordinates"] for coord in ring)
max_lon = max(coord[0] for ring in polygon["coordinates"] for coord in ring)
min_lat = min(coord[1] for ring in polygon["coordinates"] for coord in ring)
max_lat = max(coord[1] for ring in polygon["coordinates"] for coord in ring)
```
Uses only `requests` (already a dependency). No `shapely` required.

**GAM → aquifer mapping (`gam_aquifer_map.json`):**

A curated, version-controlled JSON file in the `ckan-registration` repo mapping each `package_id` to its aquifer(s):
```json
{
  "blossom-aquifer-gam": {"aquifer_name": "Blossom", "aquifer_kind": "minor"},
  "yegua-jackson-gam":   {"aquifer_name": "Yegua-Jackson", "aquifer_kind": "minor"},
  ...
}
```

For GAMs that span multiple aquifers, `aquifer_name` is an array and the fallback bbox is the union of all named aquifer extents (computed as the outer bounds of all matching polygon coordinates). Name-substring matching on the manifest title may be used as a last-resort fallback to the curated map for new models not yet in the file, but the curated map is the primary path.

This file lives alongside `gam_manifest_overrides.json` (advisory revision #10). Its content must be populated for all 23 existing models before the fallback is tested.

**New notebook cell group:** A "Generate Manifest" section in `ckan_registration.ipynb` that calls `discover_gam_models_from_local(Path(os.environ["GAM_ROOT_DIR"]))`, writes the fresh manifest to disk for human review (fill in `twdb_page_url`, `report_url`, correct any `failed_no_spatial` or `suspicious_dis` entries), and pauses before the registration loop proceeds.

---

### Capability B — TWDB metadata enrichment (landing page + PDF report)

**Goal:** augment the LLM metadata prompt with GAM report PDF text, and register the TWDB landing page and report PDF as CKAN link resources (url-type, no byte copy) alongside the model files.

#### B1: Report PDF discovery

Two-layer approach:

1. **Manifest override (takes precedence):** If the manifest record includes a non-empty `report_url` field, that URL is used directly.
2. **Auto-discovery from landing page:** Parse `<a>` tags in the TWDB landing-page HTML; identify candidate URLs pointing to `.pdf` files or links with text containing "Final Report", "Model Report", or similar. Optionally pass the top candidates to the LLM for final selection. Store the discovered URL back to the manifest record as `report_url` for future runs.

**New function in `utils.py`:**
```python
def discover_gam_report_url(
    landing_page_html: str,
    landing_page_url: str,
    *,
    candidate_keywords: list[str] | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> str | None:
    """
    Parse landing-page HTML for PDF links matching report keywords.
    Returns the best candidate URL, or None if none found.
    LLM selection is optional; use keyword heuristics first.
    """
    ...
```

#### B2: PDF text extraction

**New module: `pdf_extract.py`** (separate from `utils.py` for independent testability and deletability)

**New dependency:** `pymupdf` (`fitz`) — required. Higher-quality text extraction than pure-Python alternatives; handles multi-column layouts and mixed content better than `pypdf`. Must be added to `requirements.txt` / `environment.yml`. `pymupdf` has system-level dependencies (bundled in the wheel for most platforms; may require `libmupdf` on some Linux environments — document in setup guide).

**Functions in `pdf_extract.py`** (moved here from utils.py draft):

```python
def fetch_pdf_to_temp(
    url: str,
    *,
    timeout: int = 120,
) -> Path:
    """
    Stream a PDF from url to a NamedTemporaryFile; return the temp path.
    Raises on HTTP error. Caller is responsible for cleanup.
    """
    ...

def extract_pdf_text_chunks(
    pdf_path: Path,
    *,
    chunk_size: int = 8500,
    max_chunks: int | None = None,
) -> list[str]:
    """
    Extract all text from pdf_path using pymupdf.
    Split into chunks of approximately chunk_size characters.
    If max_chunks is set and the total would exceed it, log a warning and
    return only the first max_chunks chunks (never silently truncate —
    always log the capped count and total chunk count).
    Returns an empty list on extraction failure (best-effort).
    """
    ...

def extract_metadata_from_chunk(
    chunk: str,
    target_fields: list[str],
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
) -> dict[str, Any]:
    """
    Ask the LLM to extract candidate values for target_fields from this chunk.
    Also extract a short summary sentence. Returns {field: value_or_null, "chunk_summary": str}.
    """
    ...

def consolidate_chunk_metadata(
    chunk_results: list[dict[str, Any]],
    landing_page_excerpt: str,
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
) -> dict[str, Any]:
    """
    Merge per-chunk candidate values and summaries into a single proposed metadata dict.
    Returns the REDUCE output dict (same structure as propose_ckan_dataset_metadata_with_llm output).
    """
    ...
```

Chunk size: ~8 500 characters. A long report (100+ pages) will produce many chunks; cost and time scale with report length. A `max_chunks` safeguard is provided to allow a configurable cap — hitting the cap must be logged explicitly, never silently.

#### B3: Enhanced LLM metadata proposal — map-reduce over PDF chunks

MAP, REDUCE, and PDF fetch functions live in `pdf_extract.py` (see B2). This section describes the orchestration in `utils.py` and the decoupled call contract.

**Updated `propose_ckan_dataset_metadata_with_llm` in `utils.py`:**

```python
def propose_ckan_dataset_metadata_with_llm(
    resource_plan,
    *,
    ...
    report_pdf_url: str | None = None,   # NEW
    chunk_size: int = 8500,
    max_chunks: int | None = None,
) -> dict[str, Any]:
```

**Decoupled contract:** This function handles ONLY the map-reduce PDF pipeline and returns the REDUCE output dict. It does **NOT** call `run_persona_metadata_loop`. The notebook/orchestrator is responsible for calling:

1. `propose_ckan_dataset_metadata_with_llm(...)` → consolidated map-reduce dict
2. `run_persona_metadata_loop(consolidated_inputs, ...)` → `PersonaLoopResult`
3. `map_to_subside_dataset_fields(proposed_metadata, gam_defaults)` → CKAN package body

**When `report_pdf_url` is provided:** the function calls `fetch_pdf_to_temp` → `extract_pdf_text_chunks` → N × `extract_metadata_from_chunk` (MAP) → `consolidate_chunk_metadata` (REDUCE). All four are imported from `pdf_extract.py`.

**When `report_pdf_url` is absent (no-PDF path — backward-compatible):** behavior is unchanged — single LLM call from landing-page text only, same output structure as today. `run_persona_metadata_loop` is NOT invoked by this function in either path.

#### B4: CKAN link resources

After the model-file upload loop, add two url-type CKAN resources per GAM dataset:

| Resource | CKAN `url_type` | URL | Name |
|---|---|---|---|
| TWDB landing page | `"url"` (external link) | `twdb_page_url` from manifest | `"twdb-landing-page"` |
| GAM report PDF | `"url"` (external link) | `report_url` (discovered or manifest) | `"gam-report-pdf"` |

CKAN url-type resources have no byte upload; only metadata fields are POSTed. Use `resource_create` with `url_type="url"` and no `upload` file part.

**New function in `utils.py`:**
```python
def create_link_resources(
    ckan_url: str,
    dataset: dict[str, Any],
    link_resources: list[dict[str, Any]],
    auth_header: str | None,
) -> list[dict[str, Any]]:
    """
    Create or update url-type CKAN resources from a list of
    {"name": ..., "url": ..., "description": ..., "format": ...} dicts.
    Resource dicts may include subside_dataset resource-level fields
    (abstract, program_area, data_contact_email, caveats_usage, categories,
    primary_tags, secondary_tags, collection_method, quality_control_level,
    spatial) per the resolved field-placement decision.
    """
    ...
```

#### B5: CKAN package type and SUBSIDE field mapping

All `package_create` and `package_patch` calls in `ckan_agent.py` must set `"type": "subside_dataset"`. This is a breaking change from the current default `"dataset"` type. Existing GAM registrations (if any were created with the default type) will not be upgraded automatically — see Risks.

The LLM metadata proposal output (from the map-reduce consolidation) must be mapped to `subside_dataset` field keys before being passed to CKAN:

| SUBSIDE field key | Source |
|---|---|
| `title` | LLM proposal |
| `notes` | LLM proposal (abstract / description) |
| `tag_string` | LLM proposal |
| `url` | `twdb_page_url` from manifest |
| `temporal_coverage_start` | LLM proposal from report/landing page |
| `temporal_coverage_end` | LLM proposal from report/landing page |
| `mint_standard_variables` | MINT enrichment from existing `utils.py` (best-effort) |
| `version` | LLM proposal or `null` |
| `author` | LLM proposal or `null` |
| `license_id` | Default (e.g., `"notspecified"`) unless LLM identifies one |
| `owner_org` | From `.env` / registration config |

GAM-specific defaults applied at registration time (not LLM-proposed):

| Field | Default | Note |
|---|---|---|
| `collection_method` | `"Model Output"` | Hard-coded for all GAM registrations; controlled-vocab value |
| `categories` | `["Groundwater"]` | Default; `"Natural Hazards"` may be added for subsidence models — leave as user choice |

The `spatial` / `bounding_coordinates` field placement is subject to the open dataset-vs-resource question; its mapping is deferred until that question is resolved (see Open Questions).

---

### Capability C — Extend the subside_dataset scheming schema to the full SUBSIDE spec

**Goal:** edit `subside_dataset.yaml` in the `ckanext-dso_scheming` extension to add the missing documented dataset fields and add `choices` for controlled-vocabulary fields. This is a change to a **shared, live schema** in a **separate repo** (`modflow-suite/ckan-docker`) and requires redeployment to `ckan.tacc.utexas.edu`.

**Deployment gate:** This capability must not be deployed without explicit user approval. It is an external write affecting a production system and a shared schema used by non-GAM subside datasets. Any deployment must go through the `ckan-publisher` / deployment approval gate.

#### C1: New dataset-level fields to add

The following fields are in the SUBSIDE wiki spec (28-field target) but absent from the deployed YAML. All confirmed as `dataset_fields`.

**New descriptive fields (no controlled vocab):**

| Field key | Label | Type / preset |
|---|---|---|
| `publishing_status` | Publishing Status | `select` (see choices below) |
| `coordinate_system` | Coordinate System | free text |
| `coordinate_field_format` | Coordinate Field Format | free text |
| `collection_method_description` | Collection Method Description | `markdown` |
| `quality_assurance_description` | Quality Assurance Description | `markdown` |
| `update_frequency` | Update Frequency | `select` (see choices below) |
| `update_type` | Update Type | `select` (see choices below) |
| `from_date` | From Date | `date` |
| `to_date` | To Date | `date` |
| `disclaimer` | Disclaimer | `markdown` |
| `additional_information` | Additional Information | `markdown` |
| `related_resources` | Related Resources | `multiple_text` |
| `supporting_url` | Supporting URL | URL |

**Classification and spatial fields MOVED from resource level to dataset level (OQ-1 resolved):**

The following fields exist in the deployed YAML at `resource_fields`. They must be **removed from `resource_fields` and added to `dataset_fields`** in the extended schema. This is a structural change to the shared schema — existing `subside_dataset` datasets that populated these at resource level will need migration (see Risks).

| Field key | Label | Type / preset | Controlled vocab |
|---|---|---|---|
| `program_area` | Program Area | free text (add `choices` per wiki) | (values TBD from wiki) |
| `data_contact_email` | Data Contact Email | email | — |
| `caveats_usage` | Caveats and Usage | free text | — |
| `categories` | Categories | `multiple_select` | Boundaries, Groundwater, Natural Hazards, Planning, Water Quality, Water Use (and others per wiki) |
| `primary_tags` | Primary Tags | free text | — |
| `secondary_tags` | Secondary Tags | free text | — |
| `collection_method` | Collection Method | `select` | Administrative Record, Instrumentation Measurement, Imagery, Human Collected Observation, Survey, Model Output, Geocoding, Digitization, Ground Survey, Analysis or Synthesis, GPS Measurement, Unknown, Other |
| `quality_control_level` | Quality Control Level | `select` | (values TBD from wiki `metadata-schema.md`) |
| `spatial` | Bounding Coordinates / Spatial | `json_object` | — |

Note: the current `ckan-registration` code already writes `spatial` at dataset level — this change aligns the schema with the code's existing behavior.

Note: `uuid` is CKAN-native (the `id` field); no custom field needed. `name` / URL slug is `dataset_slug` preset (already present).

#### C2: `choices` additions and `mint_standard_variables` placement change

**Controlled-vocab `choices` to add to existing and newly-dataset-level fields:** See the tables in C1 for the `categories`, `collection_method`, `quality_control_level`, and `program_area` choices.

**`mint_standard_variables` MOVED from dataset level to resource level (OQ-2 resolved):**

The deployed YAML places `mint_standard_variables` at `dataset_fields`. It must be **removed from `dataset_fields` and added to `resource_fields`**. This matches:
- The SubsideWiki / March-21 preference for per-file MINT variable annotation.
- The existing `utils.py` functions `infer_standard_variable_names_from_resource_files` and `annotate_resource_plan_with_mint_standard_variables`, which annotate individual resources rather than the dataset as a whole.

A dataset-level aggregated union of MINT variables was considered as an alternative but is NOT adopted for this prototype (future enhancement).

**Backward-compatibility risk (both C1 and C2 changes):** Moving fields across levels and adding `choices` to free-text fields both enforce validation on save. Existing `subside_dataset` entries with resource-level classification data or with free-text values outside the controlled vocab will fail validation on next edit. **Migration must be COMPLETED AND VERIFIED before the schema PR merges or deploys.** This migration affects all `subside_dataset` entries, not just GAM datasets.

#### C3: Field-placement summary (resolved)

| Field | Old level (deployed YAML) | New level (extended schema) |
|---|---|---|
| `program_area` | resource | **dataset** |
| `data_contact_email` | resource | **dataset** |
| `caveats_usage` | resource | **dataset** |
| `categories` | resource | **dataset** |
| `primary_tags` | resource | **dataset** |
| `secondary_tags` | resource | **dataset** |
| `collection_method` | resource | **dataset** |
| `quality_control_level` | resource | **dataset** |
| `spatial` | resource | **dataset** (aligns with existing ckan-registration code behavior) |
| `mint_standard_variables` | dataset | **resource** |
| `abstract` | resource | resource (unchanged) |
| `name`, `url`, `format`, `temporal_coverage_start`, `temporal_coverage_end` | resource | resource (unchanged) |

#### C4: Redeploy `ckanext-dso_scheming`

After editing `subside_dataset.yaml`, the extension must be rebuilt and redeployed to `ckan.tacc.utexas.edu`. Steps:

1. **Edit the YAML** in `modflow-suite/ckan-docker`.
2. **Validate schema locally:** run `scheming_dataset_schema_show api` locally or against a staging instance to confirm the schema loads without error.
3. **Complete migration BEFORE merge:** enumerate all existing `subside_dataset` entries; for each, move classification fields from resource level to dataset level and verify `quality_control_level` / `collection_method` values fall within the new `choices` lists (or set to `null`). Re-save each entry and confirm no validation errors. This step is a hard gate — the PR must not be merged until migration is verified complete.
4. **Commit and PR** to `modflow-suite/ckan-docker` (requires explicit user approval for the GitHub push).
5. **Redeploy the Docker stack** (requires explicit deployment approval).

**`quality_control_level` choices:** The choices for this field are listed as "values TBD from wiki `metadata-schema.md`" in C1 above. This must be resolved (actual values inserted into the YAML, or flagged as a required pre-deploy TODO blocking the PR) before Capability C is deployed.

---

### Capability D — Three-persona iterative metadata authoring and evaluation loop

**Goal:** replace the single LLM metadata-proposal call with an agentic evaluation harness where one persona authors candidate SUBSIDE metadata and two evaluator personas independently review it, looping until both agree or a round cap is reached.

**Framing:** This capability is explicitly a **prototype and evaluation harness** for a future autonomous metadata agent. The loop's transcript and convergence metrics are first-class outputs — they provide the evidence needed to assess whether the three-persona approach produces better metadata than a single-pass proposal.

**Module:** `persona_loop.py` (new file). Contains: `run_persona_metadata_loop` orchestrator, `PersonaLoopResult` / `LoopRound` / `EvaluatorResult` dataclasses, and the three persona prompt templates (`DOMAIN_EXPERT_PROMPT`, `DATA_CURATOR_PROMPT`, `DATA_SCIENTIST_PROMPT`). This module must be independently testable and independently deletable as a prototype. `utils.py` retains CKAN / MINT / manifest helpers only.

#### D1: Personas

All three personas use the same LLM endpoint (`ai.tejas.tacc.utexas.edu`, model `Meta-Llama-3.3-70B-Instruct`) unless overridden by config. Each has a distinct system prompt stored as a module-level constant in `persona_loop.py`.

**Persona 1 — Domain Expert (Author)**

Inputs: TWDB landing-page excerpt, consolidated PDF map-reduce findings (from Capability B), model file inventory (resource_plan), DIS-derived bbox/GeoJSON, the extended `subside_dataset` schema field list (dataset-level fields including the newly moved classification fields: `categories`, `collection_method`, `quality_control_level`, `spatial`, etc.), controlled-vocab choices, GAM-specific defaults (`collection_method = "Model Output"`, `categories = ["Groundwater"]`), and (on revision rounds) the evaluators' questions and recommendations from the previous round.

**Re-raise prevention (rounds 2+):** Before sending the revision prompt to the Domain Expert, inject the prior round's resolved `_gap` annotations as an explicit section: `"Previously resolved gaps — do not re-raise: [field: reason, ...]"`. This section is derived from the prior round's `candidate_metadata` `_gap` fields. Each evaluator's prompt for round 2+ must also include this section so evaluators do not re-raise gaps that the author already marked as null/not-available.

Note: `mint_standard_variables` is now resource-level (OQ-2 resolved) and is populated by the existing `annotate_resource_plan_with_mint_standard_variables` function prior to calling the persona loop; the loop receives the annotated resource plan and does not need to propose MINT variables for the dataset level.

Output: a schema-conforming candidate metadata object with `subside_dataset` field keys. Fields that cannot be determined from the available sources must be set to `null` with a `"_gap": "reason"` annotation — the author must never fabricate values. This is the same anti-hallucination guard as the existing LLM prompt.

**Persona 2 — Data Curator (Evaluator — FAIR)**

Evaluates the candidate metadata against FAIR principles:
- Findable: persistent identifiers referenced, rich/searchable metadata, tags/categories populated, title descriptive.
- Accessible: resource URLs resolve (or are marked as link-type), format/protocol declared, data contact provided.
- Interoperable: controlled-vocab terms used (not free-text alternatives), standard formats, MINT variables present if applicable, ISO-8601 temporal fields, CRS declared with spatial.
- Reusable: license present, provenance/lineage traceable, caveats/usage populated, notes/abstract sufficient for reuse.

Output: `{"verdict": "pass" | "revise", "questions": [...], "recommendations": [...]}`. A verdict of `"pass"` means no blocking FAIR issues remain. Non-blocking suggestions are listed but do not prevent pass.

**Persona 3 — Data Scientist (Evaluator — Usability)**

Evaluates whether a domain-knowledgeable data scientist could understand and use the data without any further context:
- Abstract/description answers: what is this model, what does it represent, what geographic/temporal scope.
- Variables and units explained; acronyms expanded (e.g., "GAM" first use).
- Temporal extent (from_date, to_date) clearly stated.
- Spatial extent clearly stated and tied to a named aquifer/region.
- File roles and formats understandable from the resource names and descriptions.
- No unexplained jargon.

Output: same structure as Data Curator: `{"verdict": "pass" | "revise", "questions": [...], "recommendations": [...]}`.

**Critical guard for evaluators:** when an evaluator asks for information not present in the source material, the Domain Expert is expected to annotate the field as `null` / `"not available in sources"`. Evaluators must accept this as a valid resolution and not re-raise the same question. This prevents the loop from forcing hallucinated content.

#### D2: Loop logic and convergence

```python
def run_persona_metadata_loop(
    consolidated_inputs: dict[str, Any],
    resource_plan: list[dict[str, Any]],
    mint_standard_variables: list[str] | None,
    bbox_geojson: str | None,
    subside_schema_fields: list[dict[str, Any]],
    gam_defaults: dict[str, Any],
    *,
    max_rounds: int = 3,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
) -> PersonaLoopResult:
    """
    Runs the three-persona authoring + evaluation loop.
    Returns PersonaLoopResult with proposed_metadata, converged,
    rounds_to_converge, and full transcript.
    """
    ...
```

**Loop per round:**

1. Domain Expert authors (round 1) or revises (rounds 2+) the candidate metadata, incorporating evaluator feedback from the previous round. On rounds 2+, the prompt includes the "Previously resolved gaps — do not re-raise" section (see D1).
2. Data Curator and Data Scientist evaluate the candidate **in parallel** using `concurrent.futures.ThreadPoolExecutor` (two independent LLM calls submitted concurrently, same candidate metadata). Tests must handle non-deterministic call ordering.
3. If both return `verdict = "pass"`: loop converges; return result.
4. If either returns `verdict = "revise"`: Domain Expert revises in the next round.
5. If `round == max_rounds` and not converged: stop. Mark result as `converged = False`. Return the latest candidate metadata plus all outstanding questions/recommendations. **Never silently accept unconverged metadata.** Log the cap hit clearly.

**LLM failure handling:** `run_persona_metadata_loop` must catch ALL LLM call failures (network error, timeout, non-200 response, JSON parse failure) for any persona in any round. On failure: log the failure with the round number and persona name; return `PersonaLoopResult(converged=False, proposed_metadata=<best-available prior state or empty dict>, transcript=<incomplete transcript so far>)`. The exception must NOT propagate out of `run_persona_metadata_loop`. The outer 23-model registration loop must continue to the next model uninterrupted.

#### D3: Structured output types

```python
@dataclass
class EvaluatorResult:
    verdict: Literal["pass", "revise"]
    questions: list[str]
    recommendations: list[str]

@dataclass
class LoopRound:
    round_number: int
    candidate_metadata: dict[str, Any]        # subside_dataset field keys
    fair_evaluator: EvaluatorResult
    usability_evaluator: EvaluatorResult
    converged: bool

@dataclass
class PersonaLoopResult:
    proposed_metadata: dict[str, Any]          # final candidate (last round)
    converged: bool
    rounds_to_converge: int | None             # None if did not converge
    transcript: list[LoopRound]
    model_id: str
    timestamp: str                             # ISO-8601
```

#### D4: Audit trail

After each model run, serialize `PersonaLoopResult` to:
```
runs/<model_id>_<YYYYMMDD_HHMMSS>.json
```

The `runs/` directory sits alongside the manifest in the `ckan-registration` repo. It is the primary evaluation output for assessing the prototype. The directory is created at startup via `Path("runs/").mkdir(parents=True, exist_ok=True)` — no manual setup required. Content per file: all rounds' candidate metadata, each evaluator's verdict and questions, and the convergence outcome. Summary metrics across a full 23-model run should be computable from these files (mean rounds to converge, convergence rate, most common evaluator question categories).

`runs/` must be listed in `.gitignore` (see Files / Components table).

#### D5: Human-in-the-loop gate

The persona loop produces a **proposal only**. It does not trigger any CKAN write. The proposed metadata flows to the existing dry-run checkpoint in the notebook. If the loop did not converge, the notebook cell displays the outstanding questions prominently for human resolution before the user can proceed to `apply`.

#### D6: Cost and throttling note

Up to 3 rounds × (1 author + 2 evaluators) = up to 9 LLM calls per model for the persona loop, on top of the N map-reduce calls from Capability C. For a 23-model full run with 6 report chunks per model: (6 MAP + 1 REDUCE + up to 9 persona) × 23 = up to ~368 LLM calls.

**Inter-call sleep (throttling):** A configurable delay (`LLM_CALL_DELAY_SECONDS`, default 1 second) is applied with `time.sleep` between successive LLM/API calls. Applied between: (a) persona-loop calls (after each author and evaluator call, and between rounds) and (b) report map-reduce per-chunk calls (Capability B). May optionally be applied between CKAN API calls if rate limits are observed, but LLM calls are the primary target.

**Wall-clock tradeoff:** With a 1-second delay and ~368 LLM calls per full run, the sleep alone adds ~6 minutes of wall time (368 × 1s) across 23 models — roughly 16 seconds per model in throttle delay on top of actual LLM response time. This is a conscious tradeoff for responsible endpoint usage. Reduce the delay only if the LLM endpoint has confirmed no rate limits.

Set `max_rounds` and `max_chunks` based on acceptable cost/time budget. These are configurable; the defaults (3 rounds, no chunk cap, 1-second delay) are the conservative starting points.

---

## Files / Components Likely Affected

**`ckan-registration` repo (primary):**

| File | Change |
|---|---|
| `utils.py` | Add: `discover_gam_models_from_local`, `discover_gam_report_url`, `create_link_resources`. Extend: `propose_ckan_dataset_metadata_with_llm` (add `report_pdf_url`, `chunk_size`, `max_chunks` params; returns REDUCE output dict only — does NOT call persona loop). Add SUBSIDE field-mapping helper: `map_to_subside_dataset_fields(...)`. Keep: CKAN / MINT / manifest helpers. Remove from utils: PDF pipeline functions (moved to `pdf_extract.py`); persona-loop orchestrator and prompt templates (moved to `persona_loop.py`). |
| `pdf_extract.py` (new) | PDF pipeline module: `fetch_pdf_to_temp`, `extract_pdf_text_chunks`, `extract_metadata_from_chunk`, `consolidate_chunk_metadata`. Independently testable and independently deletable. |
| `persona_loop.py` (new) | Persona loop module: `run_persona_metadata_loop` orchestrator; `PersonaLoopResult`, `LoopRound`, `EvaluatorResult` dataclasses; prompt templates `DOMAIN_EXPERT_PROMPT`, `DATA_CURATOR_PROMPT`, `DATA_SCIENTIST_PROMPT`. Independently testable and independently deletable prototype. |
| `gam_aquifer_map.json` (new) | Curated mapping from `package_id` to `{aquifer_name, aquifer_kind: "major" \| "minor"}`. Multi-aquifer GAMs store an array of entries; the fallback chain unions their extents. Human-authored; version-controlled. |
| `gam_manifest_overrides.json` (new) | Human-authored, version-controlled overrides file. `discover_gam_models_from_local` merges this as a final step: any field present in the overrides for a `package_id` takes precedence over auto-discovered values. Enables stable corrections without regenerating the full manifest. |
| `ckan_agent.py` | Change: set `"type": "subside_dataset"` in all `package_create` / `package_patch` calls; apply GAM defaults (`collection_method`, `categories`) before CKAN write. Fix stale module docstring ("n8n-facing"). |
| `ckan_registration.ipynb` | Add: manifest generation cell group (Capability A); link-resource registration cells (Capability B); persona-loop result review cells (Capability D). Retain existing local-path registration cell group. Notebook calls map-reduce, persona loop, and field mapping as three explicit sequential steps. |
| `twdb_gam_modflow_locations_with_bbox_strings.json` | Schema extension: add `report_url`, `bbox_derivation_status` fields. Spatial fields now produced by DIS/flopy derivation (with aquifer fallback). Remove `tapis_system_id`, `tapis_root_path` (no Tapis Files). |
| `.env.sample` | Add: `GAM_ROOT_DIR`, `MODFLOW_DEFAULT_EPSG` (optional), `LLM_CALL_DELAY_SECONDS` (default 1). Remove: `TAPIS_SYSTEM_ID` (Tapis Files access dropped). |
| `.gitignore` (new) | Add entries: `runs/`, `.env`, `*.pyc`, `__pycache__/`. Security note: `.env` is currently present in the repo with no `.gitignore` — this is a credentials-exposure risk that must be resolved before any further commits. |
| `requirements.txt` / `environment.yml` | Add: `pymupdf` (required for PDF extraction); `flopy` (required for DIS-based bbox derivation). Remove: `pypdf` (was previously proposed but not yet added). |
| `runs/` (new directory) | Audit-trail storage: one JSON file per model per run, containing the full persona-loop transcript (each round's candidate metadata, evaluator verdicts, revisions) and summary metrics. Created at runtime via `Path("runs/").mkdir(parents=True, exist_ok=True)`. Listed in `.gitignore`. |

**`modflow-suite/ckan-docker` repo (separate — Capability C, deployment gated):**

| File | Change |
|---|---|
| `src/ckanext-dso_scheming/ckanext/dso_scheming/subside_dataset.yaml` | Add missing dataset-level fields (see Capability C1); add `choices` to controlled-vocabulary fields (see C2); resolve field placement for classification fields (see C3). **Deployment approval required before merge/push.** |

---

## API / Schema Changes

### Tapis authentication (unchanged)

`get_tapis_token` continues to POST to `portals.tapis.io/v3/oauth2/tokens` for CKAN authentication. No Tapis Files API calls are added or used by this feature.

### Manifest schema additions

New fields per model record (produced by `discover_gam_models_from_local`):

| Field | Type | Description |
|---|---|---|
| `report_url` | string or null | PDF report URL (empty on generation; populated by TWDB enrichment or manual edit) |
| `bbox_derivation_status` | string | `"ok_from_dis"`, `"suspicious_dis"`, `"ok_from_aquifer"`, or `"failed_no_spatial"` |

`bbox_derivation_status` values:
- `ok_from_dis` — DIS bbox derived via flopy and passed Texas-bounds sanity check.
- `suspicious_dis` — DIS bbox was rejected (xoff/yoff both zero, or center outside lat 25–37 / lon −107..−93); aquifer fallback was attempted.
- `ok_from_aquifer` — DIS rejected; bbox taken from TWDB aquifer polygon (Capability A2 fallback).
- `failed_no_spatial` — Both DIS derivation and aquifer fallback failed; spatial fields are null.

Existing spatial fields (`boundary_bbox_wgs84`, `boundary_bbox_geojson`, `dataset_spatial`) are retained but now populated by the fallback chain rather than by an external one-time process. Tapis-specific fields (`tapis_system_id`, `tapis_root_path`) are not added.

### STAC API (Capability A2 — aquifer-boundary fallback)

- **Endpoint:** `https://stacapi.pods.portals.tapis.io/api/v1`
- **Collection:** `subside-context`
- **Items used:** `major-aquifers`, `minor-aquifers`
- **Asset:** `asset["service"]["href"]` on each STAC item points to the TWDB ArcGIS FeatureServer URL (read from the STAC item; not hardcoded).
- **ArcGIS query:** fetch statewide GeoJSON ONCE per run (all aquifer features) and cache in memory. Bbox computed as min/max over polygon coordinates (no shapely dependency).
- **Aquifer name field (RESOLVED 2026-06-25):** use `AQU_NAME` (string) on the `Minor_Aquifers` layer (confirmed via `FeatureServer/0?f=json`; alternates `AQ_NAME_UL` string, `AQUIFER` int code). Match case-insensitively. Verify the major-aquifers layer shares the field when implementing.
- **Curated map:** `gam_aquifer_map.json` maps `package_id` to `{aquifer_name, aquifer_kind: "major" | "minor"}`. Multi-aquifer GAMs store an array; the fallback unions all matched extents. The map is human-authored and version-controlled.

### `propose_ckan_dataset_metadata_with_llm` signature change

New optional parameters: `report_pdf_url: str | None = None`, `chunk_size: int = 8500`, `max_chunks: int | None = None`. Backward-compatible (defaults preserve existing behavior when `report_pdf_url=None` — single LLM call, same output structure, persona loop NOT invoked).

**Decoupled contract:** this function returns the REDUCE output dict only. It does NOT call `run_persona_metadata_loop`. The notebook/orchestrator calls map-reduce, persona loop, and SUBSIDE field mapping as three explicit sequential steps. This separation makes the PDF pipeline independently testable without the persona loop.

### CKAN link resources (new resource_create calls, url-type)

Each GAM dataset gains up to two additional CKAN resources with `url_type="url"`. No new CKAN API endpoints are needed; this uses the existing `resource_create` endpoint without a file upload.

### CKAN package type change

All `package_create` and `package_patch` calls must include `"type": "subside_dataset"`. This requires the `subside_dataset` schema to be live on the target CKAN instance (Capability C prerequisite). The CKAN API shape is unchanged; only the `type` field and the set of valid extra field keys change.

### `subside_dataset.yaml` schema additions (Capability C)

New ckanext-scheming fields added to `subside_dataset.yaml` in the `modflow-suite/ckan-docker` repo. See Capability C1/C2 for the full field list. This is a change to a live shared schema; backward compatibility and migration must be addressed before deployment.

---

## Data Flow

### End-to-end path (runs on Lonestar6 / any host with files mounted)

```
[Setup — once]
  Set GAM_ROOT_DIR in .env  (path to /corral-repl/tacc/.../twdb_gam_collection)
  Tapis CKAN auth: get_tapis_token(username, password) → JWT (unchanged)
  Path("runs/").mkdir(parents=True, exist_ok=True)
       │
       ▼
[Capability A — Manifest generation (on demand)]
  Path(GAM_ROOT_DIR).rglob("*")
      → local recursive walk; detect .nam files to find model dirs
  for each model: open local .dis file → flopy reads DIS → derive bbox
  [Capability A2 — aquifer-boundary bbox fallback]
    If DIS ok (xoff/yoff non-zero AND center inside lat 25-37 / lon -107..-93):
        bbox_derivation_status = "ok_from_dis"
    Else (reject DIS → bbox_derivation_status = "suspicious_dis"):
        Fetch STAC collection "subside-context" (once per run, cached)
        Read asset["service"]["href"] from major-aquifers / minor-aquifers items
        Fetch statewide ArcGIS GeoJSON (once per run per aquifer kind, cached)
        Look up model in gam_aquifer_map.json → aquifer_name(s)
        Filter GeoJSON features by aquifer_name on AQU_NAME (case-insensitive)
        Compute bbox as min/max over polygon coords (no shapely)
        Multi-aquifer GAM: union extents of all matched features
        If matched:   bbox_derivation_status = "ok_from_aquifer"
        Else:         bbox_derivation_status = "failed_no_spatial"; spatial = null
  Merge gam_manifest_overrides.json (final step — overrides take precedence per package_id)
      → bbox_derivation_status: ok | failed_no_dis | failed_no_crs | failed_error
  discover_gam_models_from_local(root_path)
      → write FRESH twdb_gam_modflow_locations_with_bbox_strings.json (overwrites prior)
      → HUMAN REVIEW / EDIT (twdb_page_url, report_url, flagged bboxes)
       │
       ▼
[Per-model registration loop — Capabilities A + B + D]
  For each model record in manifest:
    │
    ├─ [Capability B — TWDB enrichment]
    │    fetch_source_metadata(twdb_page_url)
    │        → landing-page HTML → title + 6 000-char excerpt
    │    discover_gam_report_url(html)  OR  use manifest report_url
    │        → report_pdf_url
    │
    ├─ [File inventory — local read (unchanged from current code)]
    │    list_resource_files(Path(model["package_folder"]))
    │        → local file paths via Path.rglob
    │    build_resource_plan(files, root, source_url)
    │        → resource_plan with local_path entries
    │
    ├─ [Capability B — PDF map-reduce extraction]
    │    fetch_pdf_to_temp(report_pdf_url) → temp_pdf
    │    extract_pdf_text_chunks(temp_pdf, chunk_size=8500)
    │        → [chunk_1, ..., chunk_N]  (logged if max_chunks cap hit)
    │    MAP: extract_metadata_from_chunk(chunk_i) × N LLM calls  [sleep between calls]
    │        → candidate_i {field: value_or_null, chunk_summary: str}
    │    REDUCE: consolidate_chunk_metadata(candidates, landing_excerpt)  [sleep before call]
    │        → consolidated_inputs (source material for Capability D)
    │
    ├─ [Annotate resources — explicit step before persona loop]
    │    annotate_resource_plan_with_mint_standard_variables(resource_plan)
    │        → resource_plan with per-resource mint_standard_variables populated
    │
    ├─ [Capability D — Persona metadata loop  (persona_loop.py, called explicitly)]
    │    run_persona_metadata_loop(
    │        consolidated_inputs,   ← from B REDUCE output above (not from propose_ckan_dataset_metadata_with_llm)
    │        resource_plan,         ← annotated with per-resource MINT variables
    │        bbox_geojson,
    │        subside_schema_fields,
    │        gam_defaults={collection_method: "Model Output", categories: ["Groundwater"]},
    │        max_rounds=3,
    │    )
    │    Loop (up to 3 rounds):  [sleep between each LLM call]
    │        Round 1: DOMAIN EXPERT authors candidate_metadata
    │        Rounds 2+: inject "Previously resolved gaps — do not re-raise" into all prompts
    │        [parallel via ThreadPoolExecutor]:
    │            DATA CURATOR (FAIR) evaluator → {verdict, questions, recommendations}
    │            DATA SCIENTIST (usability) evaluator → {verdict, questions, recommendations}
    │        if both verdict == "pass": CONVERGED → exit loop
    │        else: Domain Expert revises addressing outstanding questions
    │        if round == 3 and not converged: STOP, flag as unconverged
    │    On ANY LLM failure: log(round, persona), return PersonaLoopResult(converged=False, ...)
    │        → outer 23-model loop continues uninterrupted
    │    → PersonaLoopResult{proposed_metadata, converged: bool, rounds: int, transcript: [...]}
    │    persist transcript → runs/<model_id>_<timestamp>.json
    │
    ├─ [SUBSIDE field mapping  (third explicit step after persona loop)]
    │    map_to_subside_dataset_fields(proposed_metadata, gam_defaults)
    │        → package_body {type: "subside_dataset", ...subside fields}
    │
    ├─ [Human review checkpoint]
    │    display proposed_metadata + convergence status + outstanding questions
    │    (unconverged metadata flagged clearly; human may edit before proceeding)
    │
    ├─ [Dry-run diff] → CKAN package_show → field-by-field diff
    │    Dry-run output must surface url-type link resources by name + URL,
    │    not only the package body fields (so the user can verify landing-page
    │    and PDF link resources before apply)
    │
    ├─ [Apply — guarded by approval="REGISTER"]
    │    create_or_update_ckan_dataset(type="subside_dataset", ...)
    │    upsert_resources(...)   ← local_path.open("rb") byte-upload (unchanged)
    │    create_link_resources(  ← url-type resources with subside resource fields
    │        twdb_landing_page_resource,
    │        gam_report_pdf_resource,
    │    )
    │
    └─ [Post-registration]
       log registration result per model (success / skip / error)
       (manifest is not mutated post-registration; re-run discovery to refresh)
```

---

## Risks and Tradeoffs

**Run-environment constraint (accepted design decision):** The tool must run on Lonestar6 or another host with the Corral GAM files locally mounted. This is the same constraint as the current `ckan_registration.ipynb`. It is an accepted design decision, not a risk to mitigate. Users who cannot run on LS6 cannot use the local-walk path; Tapis Files access was considered and dropped (see Decisions).

**Outbound network required:** The run host must have outbound internet access for TWDB page/PDF fetches, LLM calls, and CKAN API writes. A Lonestar6 login node satisfies this. An isolated compute node does not.

### High risk

1. **Bbox derivation from MODFLOW DIS is best-effort and will fail for most models without external georeferencing data.** The DIS file provides grid dimensions (NROW, NCOL, DELR, DELC) but typically does NOT encode the grid origin (xoff/yoff), rotation, or CRS/EPSG. These vary per model and are frequently in companion files or TWDB documentation. Derivation status will be recorded per-model (`bbox_derivation_status`); models with failed derivation are registered with `null` spatial fields. Downstream map search and spatial filtering in CKAN will be degraded for these models. This is an accepted limitation; the tool flags but does not block on missing spatial data.

3. **Map-reduce LLM cost scales with report length.** A 100-page GAM report at ~500 chars/page produces ~6 chunks at 8 500 chars each — 6 MAP calls + 1 REDUCE call = 7 LLM calls per model. Some TWDB reports exceed 300 pages. With 23 models, total LLM calls in a full registration run could reach 100–200+. At typical LLM API costs, this may be significant. The `max_chunks` safeguard provides a configurable cap; hitting it is logged explicitly. Users should set `max_chunks` based on their cost tolerance.

### Medium risk

4. **Download volume and time via Tapis.** MODFLOW GAM packages can contain hundreds of files. Downloading all bytes through Tapis (rather than reading a local mount) multiplies network overhead: one Tapis API call per file, plus HTTP transfer. Large models (e.g., the Yegua-Jackson package has dozens of MODFLOW binary output files) may take minutes. Consider: skip binary output files (`.hds`, `.cbb`, `.res`) from the download loop while still registering them as resources using metadata-only entries, then batch download on demand.

5. **PDF extraction quality varies with report format.** TWDB report PDFs vary (scanned images, multi-column layouts, appendices). `pymupdf` handles most machine-generated PDFs well but may return empty or garbled text for scanned/image-only pages. `extract_pdf_text_chunks` must handle extraction failures gracefully (return empty list, log a warning) rather than raising; the pipeline continues without PDF enrichment when extraction yields no text.

6. **Report auto-discovery misidentification.** The landing-page link heuristic may select the wrong PDF (a figure, appendix, or unrelated document). The manifest `report_url` field takes precedence when set manually. The auto-discovery result should be surfaced for user review before being written to the manifest.

7. **Manual edits to manifest are lost on regeneration.** Since discovery always writes a fresh manifest, any `twdb_page_url`, `report_url`, or corrected bbox values added manually after a previous run will be overwritten. Users must re-apply or re-run enrichment after each discovery run. This is an accepted tradeoff for implementation simplicity (see Decisions).

8. **Shared schema backward compatibility.** Adding `choices` to free-text resource-level fields (`categories`, `collection_method`, `quality_control_level`, `program_area`) enforces validation on save. Any existing CKAN dataset of type `subside_dataset` that has a free-text value not matching a choice will fail validation on next edit. This affects all subside datasets, not just GAM. A migration plan (enumerate existing values, map or grandfather) is required before the schema extension is deployed. This is a **high-risk deployment gate** for Capability C.

9. **Package type migration for existing GAM registrations.** If any GAM datasets were previously registered with `type = "dataset"` (the default), they will not be recognized as `subside_dataset` by ckanext-scheming until re-registered or migrated. The current `ckan_agent.py` does not set a type; all previously created packages are assumed to use the default type. A one-time migration script or re-apply run may be needed.

10. **Persona loop may cycle on subjective disagreements.** The evaluators and author may enter repetitive disagreement on fields that cannot be resolved from the available sources (e.g., an evaluator requests a license statement that TWDB does not publish). The 3-round cap and the "not available in sources" guard mitigate this, but the user should expect unconverged outcomes for some models and plan to resolve them manually.

11. **Single LLM for all three personas introduces correlated blind spots.** The same Llama 3.3 70B model plays all three roles. If the model systematically misunderstands a SUBSIDE field or FAIR principle, all three personas will share that blind spot. This is a known limitation of the prototype and a key metric to evaluate (do the evaluators catch errors the author made?).

12. **Total LLM call budget across Capabilities B and D.** See Capability D6 for the full call count analysis. For a 23-model run: up to ~368 LLM calls plus throttle delays. `max_rounds` and `max_chunks` are the primary controls; `LLM_CALL_DELAY_SECONDS` (default 1s) controls throttling.

### Low risk

13. **Temp file cleanup on failure.** If the registration loop raises mid-run, temp files downloaded from Tapis (model files, DIS files, report PDFs) may not be cleaned up. Use `tempfile.TemporaryDirectory` as a context manager for each model's download batch, or implement a finally-block cleanup.

14. **CKAN link resource idempotency.** `create_link_resources` must check whether a resource named `"twdb-landing-page"` or `"gam-report-pdf"` already exists (via `existing_resources_by_name`) and call `resource_update` rather than `resource_create`, matching the existing upsert behavior for file resources.

15. **Cross-repo coordination for Capability C.** The YAML edit is in `modflow-suite/ckan-docker`, which is a separate repo from `ckan-registration`. Changes to the schema must be coordinated with any team members using `subside_dataset` for non-GAM datasets. A PR review and deployment window must be scheduled.

16. **Multi-aquifer GAM bbox union may be coarser than the model extent.** Some GAMs cover only part of an aquifer. Unioning the full aquifer polygon extents will produce a wider bbox than the model actually covers. This is acceptable for initial registration and will be flagged in the audit trail; a future enhancement could use the model domain polygon if published by TWDB.

17. **ArcGIS aquifer-name field — RESOLVED 2026-06-25.** Confirmed as `AQU_NAME` (string) on the `Minor_Aquifers` layer via `FeatureServer/0?f=json`. Residual care: match case-insensitively (a wrong/mismatched name silently returns zero features → falls through to `failed_no_spatial`), and verify the major-aquifers layer shares the field name when implementing.

18. **`.env` credentials exposure (security finding).** The `.env` file is currently present in the `ckan-registration` repo with no `.gitignore`. If the repo is ever pushed to GitHub or another remote, Tapis credentials and LLM API keys will be exposed. A `.gitignore` adding `.env` must be committed before any remote push. This is a required action, not merely advisory.

---

## Alternatives Considered

- **Tapis Files API for remote data access.** Access model files over `GET /v3/files/ops/{systemId}/...` and `/content/...` to eliminate the Lonestar6 / local-mount requirement. Designed in an earlier draft. Dropped by user decision: the tool runs on Lonestar6 where files are already mounted; the added complexity and credential prerequisites of Tapis Files access are not worth the benefit. Tapis remains only for CKAN authentication. Chosen: local `Path.rglob` walk.

- **Fully-live discovery without a manifest checkpoint.** Run scan → register in a single pipeline without writing an intermediate manifest. Rejected: removes the human review gate for bboxes, TWDB URLs, and model identification. Chosen: manifest-generating discovery preserves the review checkpoint.

- **`report_url`-only in manifest, no auto-discovery.** Require users to manually populate `report_url` for all 23 models, no auto-discovery. Folded into the override approach: auto-discovery runs when `report_url` is absent; manual manifest entry takes precedence when present.

- **`pypdf` as the PDF library.** Pure Python, no system dependency. Rejected: lower extraction quality for multi-column PDFs and scanned content. Chosen: `pymupdf` as the required library; system dependency is acceptable given the target environment (TACC HPC / developer workstation, not a locked-down container).

- **Single-pass truncated extraction (hard char cap) instead of map-reduce.** Feed the first N chars of the PDF to a single LLM call. Simpler; zero extra LLM calls. Rejected: a hard truncation cap discards content from later sections of the report (methodology, model parameters, aquifer description) that contain the most useful metadata fields. Chosen: map-reduce ensures the full report is processed and findings consolidated, with a configurable chunk cap for cost control.

- **`flopy` + DIS parsing for bbox, with a mandatory external CRS config file.** Require users to supply a CRS/offset config for all 23 models before running discovery. More reliable results. Rejected: adds significant setup burden and blocks discovery for models where the CRS is unknown. Chosen: best-effort derivation with explicit status flags; null spatial does not block registration.

- **Single-persona metadata proposal (no evaluator loop).** One LLM call produces the metadata; human reviews directly. Simpler, far fewer LLM calls. Rejected for this feature: the goal is explicitly to prototype an autonomous agent evaluation harness; single-pass provides no convergence signal. Chosen: three-persona loop with audit trail.

- **Two separate LLM endpoints for author vs. evaluators.** Use a higher-capability model for the author and a faster/cheaper model for evaluation. Architecturally sound. Deferred to a future iteration; the prototype uses the same endpoint for all three personas to minimize configuration and isolate the persona-prompt variable.

- **Synchronous evaluator rounds (not parallel).** Run Data Curator first, then Data Scientist, allowing each to see the other's feedback. Could produce more aligned feedback. Rejected: parallel evaluation ensures evaluators are independent (no anchoring); the author synthesizes both in the next revision.

---

## Test Plan

Unit tests (no live CKAN required; no Tapis Files calls):

1. **`discover_gam_models_from_local` — model identification:** create a temporary local directory tree with two subdirectories each containing a `.nam` file; verify the produced manifest identifies the correct model directories and namefile paths.
2. **`discover_gam_models_from_local` — fresh manifest, no merge:** run discovery on two different fixture trees in sequence; verify the second run's manifest does not contain entries from the first run.
3. **`discover_gam_models_from_local` — bbox fallback chain:** provide a fixture with a DIS file whose xoff/yoff are both 0.0 (triggering `suspicious_dis`); mock the STAC/ArcGIS response with a valid aquifer polygon; verify `bbox_derivation_status = "ok_from_aquifer"` and that the returned bbox matches the mocked polygon extent. Also verify `bbox_derivation_status = "failed_no_spatial"` when both DIS and aquifer fallback fail; verify `bbox_derivation_status = "ok_from_dis"` when DIS passes the Texas-bounds check.

4. **Texas-bounds sanity check:** verify that a bbox whose center falls outside lat 25–37 / lon −107..−93 is rejected (produces `suspicious_dis`) and a bbox whose center falls inside those bounds is accepted (produces `ok_from_dis`). Verify the zero-origin guard independently: xoff=0.0 and yoff=0.0 → `suspicious_dis` even if the computed center would be within bounds.

5. **`gam_manifest_overrides.json` merge:** run discovery on a fixture tree; provide an overrides file with a corrected `report_url` for one model; verify the manifest entry for that model reflects the override value and other models are unaffected.
6. **`discover_gam_report_url`** — provide a saved TWDB landing-page HTML fixture (e.g., `blsm.asp`); verify the function returns a URL ending in `.pdf`.
7. **`extract_pdf_text_chunks`** — provide a small synthetic PDF (machine-generated); verify chunks are returned, each within `chunk_size` chars. Verify that when `max_chunks` is set and exceeded, a warning is logged and only `max_chunks` chunks are returned.
8. **`extract_metadata_from_chunk`** — mock the LLM call; verify the function sends the chunk text and target fields in the request; verify it returns a dict with the expected keys.
9. **`consolidate_chunk_metadata`** — mock the LLM call with two chunk-result fixtures; verify the function passes all chunk summaries and the landing-page excerpt to the LLM; verify it returns a final metadata dict.
10. **`propose_ckan_dataset_metadata_with_llm` with `report_pdf_url`** — mock `fetch_pdf_to_temp`, `extract_pdf_text_chunks`, `extract_metadata_from_chunk`, and `consolidate_chunk_metadata`; verify all four are called in the expected order and the function returns the consolidation result. Verify that `run_persona_metadata_loop` is NOT called from within this function (decoupled contract).

11b. **`propose_ckan_dataset_metadata_with_llm` no-PDF backward-compat:** call with `report_pdf_url=None`; verify: (a) same output key structure as the PDF path; (b) `run_persona_metadata_loop` is NOT invoked; (c) only a single LLM call is made.
11. **`create_link_resources`** — mock `ckan_action_post`; verify `resource_create` is called when the resource name is absent and `resource_update` is called when it exists.

12. **`run_persona_metadata_loop` — convergence:** provide mock LLM responses where both evaluators return `"pass"` on round 2; verify the function returns `converged=True` and `rounds_to_converge=2`.
13. **`run_persona_metadata_loop` — cap reached:** provide mock LLM responses where evaluators always return `"revise"`; verify the function stops after `max_rounds=3`, returns `converged=False`, and the transcript has 3 entries.
14. **`run_persona_metadata_loop` — anti-hallucination guard:** provide mock evaluator that asks for a field not in the source material; verify the author's next candidate has the field set to `null` with a gap annotation rather than an invented value.

14b. **`run_persona_metadata_loop` — LLM failure handling:** mock the Domain Expert LLM call to raise a network exception on round 1; verify: (a) the exception does not propagate; (b) the function returns `PersonaLoopResult(converged=False)`; (c) the transcript contains the partial state. Repeat for evaluator failure on round 2.

14c. **`run_persona_metadata_loop` — re-raise prevention:** run a two-round mock where in round 1 the evaluator flags a gap and the author sets it to `null` with `_gap`; verify that the round-2 evaluator prompt contains the "Previously resolved gaps — do not re-raise" section and the evaluator does not re-flag that field.

14d. **`run_persona_metadata_loop` — evaluator concurrency:** verify that the two evaluator LLM calls are submitted concurrently via `ThreadPoolExecutor` (both calls in-flight before either result is awaited). Tests must not assume a fixed call order between the two evaluators.
15. **`map_to_subside_dataset_fields`** — provide a raw metadata dict; verify output includes `"type": "subside_dataset"`, GAM defaults (`collection_method = "Model Output"`), and all expected field key mappings.
16. **Audit trail serialization:** run `run_persona_metadata_loop` with mocks; verify a `PersonaLoopResult` is serializable to JSON and the `runs/` file is written with the correct structure.
17. **SUBSIDE controlled-vocab validation:** after Capability C is deployed, call `scheming_dataset_schema_show` against the live CKAN instance; verify the returned schema includes the new fields and their `choices` arrays.

Integration / smoke test (requires live credentials):

18. **End-to-end dry-run (SUBSIDE):** run the local-walk registration for one GAM model (e.g., Blossom Aquifer) on Lonestar6 through the full pipeline — manifest generation, persona loop, field mapping, `dry-run` — against the real CKAN instance; verify `ok: true`, `type: subside_dataset` in the package body, and a valid diff table showing `subside_dataset` field keys. Do not `apply`.
19. **Schema smoke-test:** after Capability C YAML edit (pre-deploy), run `ckanext-scheming` schema validation locally; verify no unknown preset or validator errors.

---

## Documentation Plan

New documentation needed:

- `README.md` update — add a section explaining the manifest generation step (`GAM_ROOT_DIR`, Capability A), the persona-loop evaluation output in `runs/`, and the SUBSIDE schema requirement.
- `.env.sample` — add `GAM_ROOT_DIR`, `MODFLOW_DEFAULT_EPSG`, `LLM_CALL_DELAY_SECONDS` with comments.
- `ckan_registration.ipynb` cell comments — annotate manifest-generation cells, persona-loop output cells, and SUBSIDE field-mapping cells with prerequisites and expected outputs.
- `docs/subside-schema-extension.md` (new, in `modflow-suite/ckan-docker`) — documents the fields added to `subside_dataset.yaml`, controlled-vocab choices, placement decision rationale, and backward-compatibility migration notes for existing datasets.
- `docs/persona-loop-evaluation.md` (new, in `ckan-registration`) — describes the three-persona loop design, prompt templates (sanitized), how to interpret audit-trail JSON files in `runs/`, and the convergence metrics used to evaluate the prototype. **Framing note:** the loop measures convergence behavior (do the three personas reach agreement, and in how many rounds?). It is explicitly a prototype harness — it does not guarantee metadata quality improvement over a single-pass proposal, and using one model at temperature 0.3 for all three personas may produce correlated evaluations. The `runs/` audit trail provides the evidence needed to assess this.

---

## Rollout / Rollback Plan

**Rollout sequence:**

1. Implement and test Capability A (`discover_gam_models_from_local`) on Lonestar6: DIS/flopy bbox derivation, Texas-bounds sanity check, manifest output, `gam_manifest_overrides.json` merge. Verify against the existing 23-model tree; confirm manifest output matches expected structure.
2. Implement and test Capability A2 (aquifer-boundary bbox fallback): build `gam_aquifer_map.json`, filter on the confirmed `AQU_NAME` field (verify the major-aquifers layer shares it), implement STAC fetch + ArcGIS polygon lookup, verify `ok_from_aquifer` path with a GAM that has a known bad DIS origin.
3. Implement and test Capability B (PDF map-reduce enrichment + link resources + SUBSIDE field mapping `B5`); implement `pdf_extract.py` as a standalone module.
4. Implement and test Capability D (persona loop + audit trail) as `persona_loop.py`. Run against Blossom Aquifer; review transcript; evaluate convergence behavior.
5. Implement Capability B5 SUBSIDE field mapping and `package_type = subside_dataset` in `ckan_agent.py`.
6. Implement Capability C (schema extension in `modflow-suite/ckan-docker`): (a) resolve `quality_control_level` choices from wiki; (b) enumerate all existing `subside_dataset` entries and complete migration; (c) validate migration complete; (d) edit YAML, validate locally; (e) get explicit user approval; (f) PR and redeploy. **Gate: user approval required before any push or deploy. Migration must be verified complete before PR merges.**
7. After Capability C is live: smoke-test `subside_dataset` package_create with GAM defaults via dry-run.
8. Add new notebook cell groups to `ckan_registration.ipynb`; retain existing registration cell group.
9. Create `.gitignore` (add `runs/`, `.env`, `*.pyc`, `__pycache__/`) before any remote push. **Security gate: this must precede any commit that includes `.env`.**
10. Update `.env.sample`, `requirements.txt` (add `pymupdf`, `flopy`), `README.md`, and write `docs/persona-loop-evaluation.md`.

**Rollback:**
- Capabilities A, B, and D are additive; removing them does not affect the current registration pipeline (which still uses the existing local-path cells).
- Capability C (schema extension) is the highest-risk rollback scenario: if the extended YAML causes validation errors on existing datasets, the prior YAML must be restored and the Docker stack redeployed. The prior YAML should be committed under version control before any changes are made.
- Capability B5 (package type change) can be reverted by removing the `type` field from `package_create` calls; previously registered `subside_dataset` packages would need to be reviewed manually.

---

## Open Questions

None. All open questions resolved 2026-06-25. See Decisions.

---

## Decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-25 | **Byte-upload to CKAN (not register-by-reference).** Download Tapis bytes → stream to temp file → upload to CKAN. | Preserves today's byte-upload behavior; CKAN dataset is self-contained; no Tapis URL expiry risk. |
| 2026-06-25 | **Manifest-generating discovery (not fully-live).** Recursive Tapis scan produces a manifest; registration runs from the manifest as a human-reviewed checkpoint. | Preserves the human-review gate for TWDB URLs, model identification, and spatial data. |
| 2026-06-25 | **Tapis storage system must be created as an explicit prerequisite.** It does not yet exist; creation is in scope for this work. | The Tapis Files path requires a registered system ID before any code is useful. |
| 2026-06-25 | **Report PDF as CKAN link resources.** TWDB landing page URL and report PDF URL are registered as url-type CKAN resources (no byte copy). | Surfaces authoritative documentation alongside model data in CKAN; avoids mirroring large PDFs. |
| 2026-06-25 | **Report URL: auto-discover from landing page, with manifest override.** Parse anchors + optional LLM selection; manifest `report_url` field takes precedence. | Automates the common case; manual override handles exceptions and avoids re-discovery on subsequent runs. |
| 2026-06-25 | **JWT Files scope: confirmed assumption.** The password-grant JWT from `get_tapis_token` will carry Files read permission on the new Tapis system. No separate token required. | User-confirmed. Must be validated in smoke-test; treat failure as a blocker. |
| 2026-06-25 | **PDF library: `pymupdf` required.** Single required dependency for PDF text extraction. Chunk size ~8 500 chars. | Higher extraction quality than `pypdf` for multi-column and mixed-content PDFs; system-level dep is acceptable in target environment. |
| 2026-06-25 | **PDF extraction strategy: map-reduce over chunks.** Split report into ~8 500-char chunks; run one LLM MAP call per chunk (extract candidate metadata + summary); one LLM REDUCE call to consolidate. `max_chunks` cap is configurable; hitting it must be logged explicitly. | Ensures full report is processed rather than silently truncated; cap controls cost for very long reports. |
| 2026-06-25 | **Manifest generation: always write fresh, no merge.** Each discovery run overwrites the manifest. Manual edits (twdb_page_url, report_url, corrected bboxes) must be re-applied after regeneration. | Eliminates merge-collision complexity and stale-state bugs; tool is a generator, not an authoritative store. |
| 2026-06-25 | **Bbox derivation: best-effort via `flopy` + MODFLOW DIS file.** Download the `.dis` file during discovery; use `flopy` to read grid dimensions; combine with CRS/offset when available to compute WGS84 bbox. Record `bbox_derivation_status` per model; null spatial does not block registration. `MODFLOW_DEFAULT_EPSG` env var provides a fallback CRS. | DIS provides grid geometry but not reliably georeferencing; honest failure flags are preferable to silently incorrect spatial data. |
| 2026-06-25 | **Model definition: a directory subtree containing at least one MODFLOW `.nam` / `.NAM` namefile constitutes a model.** No additional criteria required at this time. | Matches the structure of all 23 existing manifest entries; .nam files are the universal MODFLOW entry point. |
| 2026-06-25 | **Target schema: full 28-field SUBSIDE spec; extend `subside_dataset.yaml` first.** GAM registrations must use `type = subside_dataset` and conform to the extended schema. The schema extension (Capability C) is a prerequisite for full Capability B5 conformance. | User-confirmed. Source of truth for the 28-field list: `subside-metadata-schema-spreadsheets.md`. |
| 2026-06-25 | **`subside_dataset` schema is live on `ckan.tacc.utexas.edu`.** No "verify it exists" step is needed; a smoke-test confirmation is still prudent before the first write. | User-confirmed. |
| 2026-06-25 | **GAM defaults: `collection_method = "Model Output"`, `categories = ["Groundwater"]`.** Applied at registration time; not LLM-proposed. `"Natural Hazards"` may be added to categories for subsidence models — left as a per-model user choice. | GAMs are model outputs; Groundwater is the unambiguous primary category. |
| 2026-06-25 | **Metadata generation uses a three-persona loop (Domain Expert author + Data Curator/FAIR + Data Scientist/usability evaluators), converge-or-3-loops, human review on non-convergence.** This feature is a prototype and evaluation harness for a future autonomous metadata agent. | Explicit user framing. Audit-trail and convergence metrics are first-class outputs. |
| 2026-06-25 | **Persona loop cap: 3 rounds** (reduced from 5). If not converged after 3 rounds, return latest candidate + outstanding questions, flagged for human review. | Cost/time control; 3 rounds is sufficient to observe convergence behavior in the prototype. |
| 2026-06-25 | **Configurable inter-call sleep (`LLM_CALL_DELAY_SECONDS`, default 1 second)** between successive LLM/API calls. Applied between persona-loop calls (author + evaluators, between rounds) and between report map-reduce per-chunk calls. | Avoids rate-limiting on shared LLM endpoint; makes the tool respectful of shared infrastructure. Tradeoff: adds wall-clock time proportional to call count × delay. |
| 2026-06-25 | **Dropped Tapis Files data access; the tool runs on Lonestar6 and recursively walks a given local directory (`GAM_ROOT_DIR`) to discover models and read files.** Tapis remains only for CKAN authentication. | User decision: the tool runs on LS6 where files are already mounted; Tapis Files adds credential/infrastructure complexity without benefit. Tapis auth (`get_tapis_token`) is unchanged. |
| 2026-06-25 | **OQ-1 resolved: classification and spatial fields move to dataset level.** `program_area`, `data_contact_email`, `caveats_usage`, `categories`, `primary_tags`, `secondary_tags`, `collection_method`, `quality_control_level`, and `spatial` are removed from `resource_fields` and added to `dataset_fields` in the extended `subside_dataset.yaml`. The existing ckan-registration code already writes `spatial` at dataset level; this change brings the schema into alignment. Migration required for existing datasets that populated these at resource level. | User-resolved OQ-1. Aligns with SubsideWiki intent; avoids duplicating dataset-wide classification on every resource. |
| 2026-06-25 | **OQ-2 resolved: `mint_standard_variables` moves to resource level.** `mint_standard_variables` is removed from `dataset_fields` and added to `resource_fields` in the extended `subside_dataset.yaml`. Matches existing per-resource MINT enrichment in `utils.py` (`annotate_resource_plan_with_mint_standard_variables`). A dataset-level aggregated union was considered but NOT adopted for the prototype (future enhancement). | User-resolved OQ-2. Wiki/March-21 preference; matches the code's existing annotation model. |
| 2026-06-25 | **OQ-3 resolved: 28-field target per `subside-metadata-schema-spreadsheets.md`.** The xlsx is the authoritative source; the 24-field wiki page count is not the target. | Informational close. |
| 2026-06-25 | **OQ-4 resolved (tunable): all three personas use temperature 0.3.** Starting default for prototype. May be differentiated in a future iteration. | Balances structured output reliability with enough variance for independent perspectives. |
| 2026-06-25 | **OQ-5 resolved (tunable): a `"pass"` verdict may include non-blocking suggestions.** Only a `"revise"` verdict (with blocking questions) restarts the loop. Non-blocking suggestions are logged in the transcript but do not trigger a revision round. | Prevents the loop from cycling on minor stylistic preferences that cannot be resolved from source material. |
| 2026-06-25 | **OQ-6 resolved (tunable): persona loop operates at dataset level only in the prototype.** Resource-level metadata evaluation (per-file name, abstract, format, spatial) is a future enhancement. | Limits scope of the prototype to the most impactful metadata tier; resource-level can be layered in once dataset-level is validated. |
| 2026-06-25 | **Module split: `persona_loop.py` and `pdf_extract.py` as separate modules.** `persona_loop.py` contains `run_persona_metadata_loop`, dataclasses, and prompt templates. `pdf_extract.py` contains the four PDF pipeline functions. `utils.py` retains only CKAN/MINT/manifest helpers. | Independent testability and independent deletability of the prototype. Team-discourse revision (blocking). |
| 2026-06-25 | **Decoupled contract: `propose_ckan_dataset_metadata_with_llm` returns REDUCE output dict ONLY; does NOT call `run_persona_metadata_loop`.** Notebook/orchestrator calls map-reduce → persona loop → field mapping as three explicit steps. | Makes PDF pipeline independently testable without persona loop. Team-discourse revision (blocking). |
| 2026-06-25 | **Evaluator concurrency: `concurrent.futures.ThreadPoolExecutor`.** Two evaluator LLM calls submitted concurrently per round. Tests must handle non-deterministic call ordering. | Reduces per-round wall-clock time; standard stdlib primitive. Team-discourse revision (blocking). |
| 2026-06-25 | **LLM failure handling: catch all LLM call failures in `run_persona_metadata_loop`; log round + persona; return `PersonaLoopResult(converged=False, proposed_metadata=best-available)`; do NOT propagate.** The outer 23-model loop continues uninterrupted. | Robustness — one model's LLM failure must not abort a 23-model run. Team-discourse revision (blocking). |
| 2026-06-25 | **Evaluator re-raise prevention: inject prior round's resolved `_gap` annotations into evaluator prompts as "Previously resolved gaps — do not re-raise" section for rounds 2+.** | Prevents infinite loop on fields unavailable in source material. Team-discourse revision (blocking). |
| 2026-06-25 | **Migration gate hardened: migration must be COMPLETED AND VERIFIED before the schema PR merges/deploys.** C4 rollout steps updated accordingly. | Prevents validation failures on existing datasets after deploy. Team-discourse revision (blocking). |
| 2026-06-25 | **Aquifer-boundary bbox fallback chain (Capability A2).** Three-step chain: (1) DIS via flopy + Texas-bounds sanity check → `ok_from_dis`; (2) TWDB aquifer polygon from STAC/ArcGIS → `ok_from_aquifer`; (3) null → `failed_no_spatial`. Status `suspicious_dis` when DIS rejected. | Provides spatial coverage for GAMs with zero-origin or non-georeferenced DIS files. Team-discourse revision (blocking + Part 2). |
| 2026-06-25 | **Bbox computation: min/max over polygon coordinates; no shapely.** | Avoids a compiled dependency for a simple bounding-box operation. Team-discourse revision. |
| 2026-06-25 | **`gam_aquifer_map.json` curated mapping; human-authored; version-controlled.** Multi-aquifer GAMs store array; union extents. | Stable, auditable linking between model IDs and TWDB aquifer names. Team-discourse revision. |
| 2026-06-25 | **`gam_manifest_overrides.json` human-authored overrides; merged as final step by `discover_gam_models_from_local`.** | Enables stable manual corrections without regenerating the full manifest. Team-discourse revision (advisory). |
| 2026-06-25 | **`.gitignore` required (new file); lists `runs/`, `.env`, `*.pyc`, `__pycache__/`.** `.env` is currently present with no `.gitignore` — security finding; must precede any remote push. | Prevents accidental credential exposure and test-output pollution. Team-discourse revision (advisory / security). |
| 2026-06-25 | **`runs/` directory created at runtime via `Path("runs/").mkdir(parents=True, exist_ok=True)`.** | No manual setup required; idempotent. Team-discourse revision (advisory). |
| 2026-06-25 | **Dry-run diff must surface url-type link resources (name + URL), not only package body fields.** | Users need to review landing-page and PDF link resources before apply. Team-discourse revision (advisory). |
| 2026-06-25 | **Docs framing: persona loop measures CONVERGENCE behavior, not guaranteed quality improvement.** Single model at temp 0.3 may produce correlated evaluations. `runs/` audit trail provides evidence for assessment. | Accurate expectation-setting for a prototype. Team-discourse revision (advisory). |

---

## User Feedback / Decisions

**2026-06-25 — Coordinator-relayed:** The coordinator reports the user reviewed the design and approved it, resolving OQ-1 (classification + spatial fields → dataset level), OQ-2 (`mint_standard_variables` → resource level), and adopting the recommended defaults for OQ-4/5/6. The coordinator also requests that Status be set to **Approved**.

**2026-06-25 — User approval (direct):** The user directly instructed "resolve OQ-1/OQ-2 now, then approve" and personally selected both resolutions: OQ-1 → classification + spatial fields at **dataset** level; OQ-2 → `mint_standard_variables` at **resource** level. Status is therefore set to **Approved**. Implementation may begin on Capabilities A, B, and D.

**Approval gates still in force (not waived by this approval):**
- Capability C's CKAN scheming-schema redeploy to `ckan.tacc.utexas.edu` is a separate external/deployment write requiring its own explicit approval at execution time.
- All CKAN `apply` operations (package create/patch, resource create/update/delete) remain separately approval-gated external writes; dry-run first.

**2026-06-25 — Post-discourse revisions incorporated (coordinator-relayed):** The coordinator relayed 14 team-discourse revisions (6 blocking, 2 required-before-Capability-C-deploy, 6 advisory) and the aquifer-boundary bbox fallback feature (Capability A2). All have been applied to this spec. Per policy, coordinator-relayed acceptance is not user confirmation of approval — these are spec updates pending direct user review. The spec remains **Approved**; the substantive additions (Capability A2, `persona_loop.py`/`pdf_extract.py` split, evaluator concurrency, LLM failure handling, re-raise prevention, migration gate hardening, security `.gitignore` finding) are recorded in the Decisions table and are ready for user review at the next checkpoint.

**Open items:**
- ArcGIS aquifer-name field: **RESOLVED 2026-06-25** → `AQU_NAME` on the `Minor_Aquifers` layer (see Risk 17 / Capability A2). Verify the major-aquifers layer shares the field at implementation time.
- `quality_control_level` choices: still must be resolved from wiki `metadata-schema.md` before Capability C can deploy (see C4). This is the one remaining pre-deploy lookup, and it gates only Capability C — not A or B.
