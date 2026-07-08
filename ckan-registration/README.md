# CKAN Registration Workflows — GAM Registration

Standalone workflows to register or refresh CKAN datasets for TWDB Groundwater
Availability Models (GAMs), implementing Capabilities A (spatial discovery),
B (PDF enrichment), and D (persona-loop metadata evaluation).

## Repository layout

```
ckan-registration/
  pyproject.toml                  # package metadata, deps, pytest config
  requirements.txt                # mirrors pyproject deps for pip install -r
  README.md
  .env  .env.sample  .gitignore
  src/gam_registration/           # Python package
    __init__.py
    aquifer.py       discovery.py   pdf_extract.py  twdb_enrich.py
    subside_mapping.py  persona_loop.py  orchestrate.py
    utils.py         ckan_agent.py
  tests/                          # pytest suite (117 tests)
  notebooks/
    ckan_registration.ipynb
    notebook_ckan_registration.ipynb
  data/                           # input JSON files (human-authored)
    gam_aquifer_map.json
    gam_manifest_overrides.json
    twdb_gam_modflow_locations_with_bbox_strings.json
  fixtures/sample_model/          # sample MODFLOW model (ygjk_tr.*)
  schema/  docs/
  archive/
    Opera- Subsidence.ipynb       # legacy notebook
```

## Installation

```bash
# Editable install (recommended for development)
pip install -e .

# Or just install deps for running notebooks/tests
pip install -r requirements.txt
```

## Running tests

```bash
pytest
```

`pyproject.toml` sets `pythonpath = ["src"]` so pytest finds the package without
an editable install.

Expected: **117 passed**.

## Running the CLI

The `ckan-agent` CLI entry point is installed by `pip install -e .`:

```bash
ckan-agent --help
```

Or without an install:

```bash
python -m gam_registration.ckan_agent --help
```

Supported commands: `analyze`, `dry-run`, `revise`, `apply`, `show`.

## Running notebooks

Open either notebook from the `notebooks/` directory. Both notebooks add the
repo's `src/` directory to `sys.path` automatically, so the `gam_registration`
package is importable without a prior `pip install`.

### `notebooks/ckan_registration.ipynb`

Manifest-driven batch registration for TWDB GAM packages:
- Loads a model manifest (default: `data/twdb_gam_modflow_locations_with_bbox_strings.json`).
- Optionally runs local discovery to generate a fresh manifest from a GAM root directory.
- Runs the full Capability A/B/D pipeline per model (discovery → PDF enrichment → persona loop → SUBSIDE mapping).
- Dry-run and apply gates before writing to CKAN.

### `notebooks/notebook_ckan_registration.ipynb`

Registers a Jupyter notebook as a CKAN dataset:
- Reads a target notebook from `NOTEBOOK_PATH`.
- Uses LLM passes to propose CKAN metadata from notebook source/output.
- Creates or updates a CKAN package.

## Configuration

1. Copy `.env.sample` to `.env` and fill in credentials.
2. Required: `CKAN_BASE_URL`, `CKAN_API_TOKEN` (or `CKAN_USERNAME`/`CKAN_PASSWORD`).
3. Optional: `OPENAI_API_KEY` (or `OPENAI_BASE_URL` for local LLMs), `GAM_ROOT_DIR`.

## Data files

The JSON files in `data/` are version-controlled human-authored inputs:

| File | Purpose |
|------|---------|
| `gam_aquifer_map.json` | Maps package IDs to aquifer names/kinds for bbox fallback |
| `gam_manifest_overrides.json` | Per-model field overrides applied after discovery |
| `twdb_gam_modflow_locations_with_bbox_strings.json` | TWDB GAM model manifest |

`discovery.py` resolves these paths relative to the repo root (via `Path(__file__).parents[1]`),
so they work correctly regardless of the working directory when imported.

## n8n chat registration agent

The `ckan_agent.py` CLI supports five commands callable from n8n or a shell:

- `analyze` — builds proposed metadata and resource state.
- `revise` — applies chat-derived edits to saved state.
- `dry-run` — compares proposed state with live CKAN.
- `apply` — writes to CKAN only when `"approval": "REGISTER"` is set.
- `show` — prints saved session state.
