"""ESS-DIVE round-trip evaluation.

For each dataset in ``tests/ess_dive_ckan_test_datasets.json``:

1. Download its resource files (smart cap: text/CSV/markdown/PDF/image up to a
   per-file byte limit; oversized or agent-unreadable types are passed as
   metadata only).
2. Hand the staged files + a resource manifest to the agent and ask it to draft
   the full CKAN package metadata.
3. Compare the agent's CKAN package to the fixture's gold ``ckan_package``
   field by field and report coverage.

Run:
    python -m basic_ckan_agent.evaluation.ess_dive_eval
    python -m basic_ckan_agent.evaluation.ess_dive_eval --dataset ess-dive-2319247-yakama-river-game-camera-stream-inundation
    python -m basic_ckan_agent.evaluation.ess_dive_eval --max-bytes 10485760 --limit 2
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

from basic_ckan_agent.evaluation.extraction import _last_json_object
from basic_ckan_agent.evaluation.ess_dive_scoring import DatasetScore, score_dataset
from basic_ckan_agent.files.safety import SUPPORTED_PATH_SUFFIXES
from basic_ckan_agent.logging_config import LOG_DIR, logger

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "ess_dive_ckan_test_datasets.json"
REPORT_DIR = LOG_DIR / "ess_dive_eval"
DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # per-file download cap

# GDAL/OGR pullability: which resource formats a GDAL-based tool could ingest.
GDAL_RASTER_EXT = {".tif", ".tiff", ".nc", ".hdf", ".h5", ".grib", ".grb", ".grib2", ".img", ".vrt", ".jp2", ".asc", ".dem", ".bil"}
OGR_VECTOR_EXT = {".shp", ".geojson", ".gml", ".kml", ".kmz", ".gpkg", ".gdb", ".tab", ".mif"}
PDAL_EXT = {".las", ".laz", ".copc"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def gdal_capability(resource: dict) -> dict:
    """Classify whether a GDAL/OGR (or PDAL) tool could pull a resource's data."""
    url = str(resource.get("url") or "")
    ext = _suffix_from_url(url)
    fmt = str(resource.get("format") or "").lower()

    if ext in GDAL_RASTER_EXT or fmt in {"netcdf", "geotiff", "tiff", "grib", "hdf", "cog"}:
        return {"pullable": "yes", "driver": "GDAL raster", "note": f"{ext or fmt} is a GDAL raster format"}
    if ext in OGR_VECTOR_EXT or fmt in {"geojson", "shapefile", "kml", "gpkg", "gml"}:
        return {"pullable": "yes", "driver": "OGR vector", "note": f"{ext or fmt} is an OGR vector format"}
    if ext in PDAL_EXT or fmt in {"las", "laz"}:
        return {"pullable": "maybe", "driver": "PDAL", "note": "point cloud — via PDAL, not core GDAL"}
    if ext in IMAGE_EXT or fmt in {"jpg", "jpeg", "png"}:
        return {"pullable": "maybe", "driver": "GDAL image", "note": "readable as raster, likely not georeferenced"}
    if ext in {".csv", ".tsv"} or fmt in {"csv", "tsv"}:
        return {"pullable": "maybe", "driver": "OGR CSV", "note": "only if it has geometry/coordinate columns"}
    if ext == ".zip" or fmt == "zip":
        return {"pullable": "maybe", "driver": "GDAL/OGR", "note": "only if archive contains a shapefile or geodatabase"}
    return {"pullable": "no", "driver": "", "note": f"{ext or fmt or 'unknown'} is not a GDAL/OGR format"}

# File-heavy turns need more tool steps than the default; give them headroom.
os.environ.setdefault("CKAN_AGENT_RECURSION_LIMIT", "24")

# Fields compared against the gold ckan_package.
@dataclass
class StagedResource:
    name: str
    resource: dict
    local_path: Path | None = None
    skipped_reason: str | None = None


@dataclass
class DatasetResult:
    name: str
    staged: list[StagedResource] = field(default_factory=list)
    generated: dict = field(default_factory=dict)
    score: DatasetScore | None = None
    answer: str = ""
    error: str | None = None


def file_source_context(staged: list[StagedResource], max_chars: int = 4000) -> str:
    """Build the free-text judge's source context from the downloaded files.

    This is the only ground truth the agent saw (no external record), so
    faithfulness is judged against the actual file contents + manifest.
    """
    parts: list[str] = []
    budget = max_chars
    for s in staged:
        if s.local_path and s.local_path.suffix.lower() in {".txt", ".md", ".csv", ".tsv", ".json"} and budget > 0:
            try:
                text = s.local_path.read_text(errors="replace")[:1200]
            except Exception:
                text = ""
            if text.strip():
                chunk = f"--- {s.name} ({s.resource.get('format')}) ---\n{text}"
                parts.append(chunk[:budget])
                budget -= len(chunk)
        else:
            parts.append(f"--- {s.name} ({s.resource.get('format')}) [not read]: {s.resource.get('description','')}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #

def download_resources(resources: list[dict], dest: Path, max_bytes: int) -> list[StagedResource]:
    dest.mkdir(parents=True, exist_ok=True)
    staged: list[StagedResource] = []
    for index, resource in enumerate(resources, start=1):
        name = str(resource.get("name") or f"resource-{index}")
        url = str(resource.get("url") or "")
        suffix = _suffix_from_url(url)

        if suffix not in SUPPORTED_PATH_SUFFIXES:
            staged.append(StagedResource(name, resource, skipped_reason=f"unsupported type '{suffix or '?'}'"))
            continue

        filename = f"{index:02d}-{_safe_basename(url, suffix)}"
        target = dest / filename
        try:
            ok, reason = _download_capped(url, target, max_bytes)
        except Exception as exc:
            logger.exception("Download failed for %s", url)
            staged.append(StagedResource(name, resource, skipped_reason=f"download error: {exc}"))
            continue

        if ok:
            staged.append(StagedResource(name, resource, local_path=target))
        else:
            target.unlink(missing_ok=True)
            staged.append(StagedResource(name, resource, skipped_reason=reason))
    return staged


def _download_capped(url: str, target: Path, max_bytes: int) -> tuple[bool, str]:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            return False, f"too large ({int(declared)} > {max_bytes} bytes)"

        written = 0
        with open(target, "wb") as fh:
            for chunk in response.iter_content(chunk_size=65536):
                written += len(chunk)
                if written > max_bytes:
                    return False, f"exceeded cap while streaming (> {max_bytes} bytes)"
                fh.write(chunk)
    return True, "ok"


def _suffix_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    return Path(path).suffix.lower()


def _safe_basename(url: str, suffix: str) -> str:
    path = unquote(urlparse(url).path)
    base = Path(path).name or "file"
    return base if base.lower().endswith(suffix) else base + suffix


# --------------------------------------------------------------------------- #
# 2. Ask the agent
# --------------------------------------------------------------------------- #

_GOLD_FIELDS_HINT = (
    "name, title, notes, url, license_id, version, type, spatial (stringified GeoJSON), "
    "temporal_coverage_start, temporal_coverage_end, tags (list of {\"name\": ...}), "
    "extras (list of {\"key\":..., \"value\":...}), resources (list of {name, description, format, mimetype, url})"
)

# What the agent is asked to produce: dataset-level metadata only. Resource URLs
# are NOT requested — they are injected from the manifest (the agent was prone to
# rewriting/mangling them), so there is nothing for it to transcribe incorrectly.
_AGENT_FIELDS_HINT = (
    "name (a short lowercase hyphenated slug), title, notes, type, tags (topical keywords); plus "
    "license_id, version, spatial, temporal_coverage_start, temporal_coverage_end ONLY if explicitly stated "
    "in the files (otherwise state they cannot be determined). Do NOT include resources or any URLs."
)


def build_request(staged: list[StagedResource]) -> str:
    readable = [s for s in staged if s.local_path]
    manifest = [
        {
            "name": s.name,
            "format": s.resource.get("format"),
            "mimetype": s.resource.get("mimetype"),
            "url": s.resource.get("url"),
            "description": s.resource.get("description"),
            "local_file": str(s.local_path) if s.local_path else None,
            "note": s.skipped_reason,
        }
        for s in staged
    ]
    paths = "\n".join(f"- `{s.local_path}`" for s in readable) or "(none readable)"

    return (
        "You are drafting CKAN dataset metadata from a set of downloaded ESS-DIVE resource files.\n\n"
        "Local files you can read with the file tools:\n"
        f"{paths}\n\n"
        "Full resource manifest (some entries are metadata-only because the file is large or a binary type "
        "the file tools cannot read):\n"
        f"{json.dumps(manifest, indent=2, ensure_ascii=False)}\n\n"
        "Inspect the readable files, then propose CKAN dataset-level metadata. Base every field only on the file "
        "contents and the manifest; do not invent facts. If a field cannot be determined from the files, say so "
        "rather than guessing. Do not call CKAN write tools.\n"
        f"Propose these fields: {_AGENT_FIELDS_HINT}.\n"
        "Requirements:\n"
        "- ALWAYS provide a name (a short lowercase hyphenated slug), a title, and notes (a 1-3 sentence "
        "description). Never leave these blank.\n"
        "- Do NOT produce a resources list and do NOT write any file URLs. The resource list is attached "
        "automatically from the manifest above; you only describe the dataset-level metadata.\n"
        "Describe each proposed field in plain prose. Do NOT output JSON or any fenced code block in your answer "
        "(this endpoint mis-parses code blocks as tool calls)."
    )


# Phase 2: a tools-free model call turns the prose proposal into JSON. With no
# tools bound, the litellm proxy does not attempt function-call extraction, so a
# JSON answer is safe (unlike a JSON block emitted during a tool-bound turn).
# Built by concatenation (not str.format) because _GOLD_FIELDS_HINT contains
# literal braces like {"name": ...} that str.format would treat as fields.
_EXTRACT_INSTRUCTION = (
    "Convert the proposed CKAN dataset metadata below into a single JSON object with exactly one key, "
    '"ckan_package", whose value is an object with these fields: ' + _GOLD_FIELDS_HINT + ".\n"
    "Use only values supported by the proposal; use null or an empty list where the proposal gives nothing. "
    "Output raw, valid JSON only — no markdown, no code fences, no comments.\n\n"
    "Proposed metadata:\n"
)


def extract_package_json(proposal: str, model_name: str | None = None) -> dict:
    """Phase 2: reformat the agent's prose proposal into a ckan_package dict."""
    from basic_ckan_agent.llm.model import build_model

    response = build_model(model_name).invoke(_EXTRACT_INSTRUCTION + proposal)
    return parse_generated_package(str(response.content))


def parse_generated_package(answer: str) -> dict:
    obj = _last_json_object(answer)
    if isinstance(obj, dict):
        pkg = obj.get("ckan_package")
        if isinstance(pkg, dict):
            return pkg
        return obj  # agent may have returned the package directly
    return {}


# --------------------------------------------------------------------------- #
# 3. Runner + report  (scoring lives in ess_dive_scoring.py)
# --------------------------------------------------------------------------- #

def load_datasets(path: Path = FIXTURE) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [d["ckan_package"] for d in data.get("datasets", []) if isinstance(d, dict) and "ckan_package" in d]


def run_dataset(package: dict, stage_root: Path, max_bytes: int) -> DatasetResult:
    from basic_ckan_agent.llm.model import build_model
    from basic_ckan_agent.runtime.graph import ChatSession

    name = str(package.get("name", "dataset"))
    result = DatasetResult(name=name)
    dataset_dir = stage_root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")

    result.staged = download_resources(package.get("resources", []), dataset_dir, max_bytes)
    n_read = sum(1 for s in result.staged if s.local_path)
    n_gdal = sum(1 for s in result.staged if gdal_capability(s.resource)["pullable"] == "yes")
    print(f"  resources: {len(result.staged)} total, {n_read} downloaded, {len(result.staged) - n_read} metadata-only")
    print(f"  GDAL-pullable resources: {n_gdal}")

    try:
        # Phase 1: agent reads files and proposes metadata in prose.
        answer = ChatSession().ask(build_request(result.staged))
        result.answer = answer
        # Phase 2: tools-free call reformats the prose proposal into JSON.
        result.generated = extract_package_json(answer)
    except Exception as exc:
        logger.exception("Agent run failed for %s", name)
        result.error = str(exc)
        return result

    if not result.generated:
        result.error = "could not parse a ckan_package from the agent reply"
        print("  WARNING: no ckan_package parsed from agent reply")
        return result

    # Inject resources from the manifest (known ground truth). The agent is no
    # longer asked for resource URLs — it cannot transcribe/mangle what it never
    # emits — so resource provenance is exact by construction and there is no
    # resource-fidelity check to run.
    result.generated["resources"] = [
        {k: r.get(k) for k in ("name", "description", "format", "mimetype", "url")}
        for r in package.get("resources", [])
        if isinstance(r, dict)
    ]

    # Files-only inputs: no external DOI record, so license/temporal/spatial/extras
    # are honestly classified "unavailable" and excluded from the derivable score.
    result.score = score_dataset(
        result.generated,
        package,
        record=None,
        record_provided=False,
        source_context=file_source_context(result.staged),
        judge_model=build_model(),
    )
    _print_dataset_report(result)
    return result


_MARK = {"match": "✓", "partial": "~", "differ": "✗", "missing": "✗", "n/a": "·"}


def _print_dataset_report(result: DatasetResult) -> None:
    s = result.score
    if not s:
        return
    print(f"  derivable metadata score: {s.derivable_score:.0%}    full catalog completeness: {s.catalog_completeness:.0%}")
    j = s.free_text_judgment
    if j:
        print(f"  free-text judge: title={j.get('title_score')}/5 notes={j.get('notes_score')}/5 "
              f"faithful={j.get('faithfulness_pass')} — {str(j.get('comment',''))[:120]}")
    for fname, fs in s.fields.items():
        if fs.availability == "unavailable":
            print(f"    · {fname:<24} excluded (not available from inputs)")
            continue
        sc = "" if fs.score is None else f" {fs.score:.2f}"
        print(f"    {_MARK.get(fs.status, '?')} {fname:<24} {fs.status}{sc}  {fs.note}")
    if s.fidelity_errors:
        print("  fidelity errors:")
        for e in s.fidelity_errors:
            print(f"    ! {e}")
    if s.gates_applied:
        print("  gates:", "; ".join(s.gates_applied))
    print("  GDAL pullability:")
    for st in result.staged:
        cap = gdal_capability(st.resource)
        glyph = {"yes": "▣", "maybe": "▤", "no": "·"}.get(cap["pullable"], "?")
        drv = f"{cap['driver']} — " if cap["driver"] else ""
        print(f"    {glyph} {st.name:<40} {cap['pullable']:<6} {drv}{cap['note']}")


def run(
    dataset_filter: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    limit: int | None = None,
    *,
    fixture: str | Path | None = None,
    label: str | None = None,
) -> list[DatasetResult]:
    """Run the eval against a fixture JSON.

    fixture: path to a datasets JSON ({"datasets": [{"ckan_package": {...}}]}).
             Defaults to the bundled ESS-DIVE fixture.
    label:   prefix for the output JSON/HTML filenames (for traceability).
             Defaults to the fixture's filename stem.
    """
    fixture_path = Path(fixture) if fixture else FIXTURE
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        return []
    out_label = _safe_label(label or fixture_path.stem)

    try:
        packages = load_datasets(fixture_path)
    except Exception as exc:
        print(f"Could not load fixture {fixture_path}: {exc}")
        return []
    if not packages:
        print(
            f"No datasets found in {fixture_path}. Expected JSON shape: "
            '{"datasets": [{"ckan_package": {"name": ..., "resources": [...]}}, ...]}'
        )
        return []

    if dataset_filter:
        packages = [p for p in packages if p.get("name") == dataset_filter]
    if limit:
        packages = packages[:limit]
    if not packages:
        print("No matching datasets in fixture.")
        return []

    print(f"Fixture: {fixture_path}  ({len(packages)} dataset(s))  ·  output label: {out_label}")
    results: list[DatasetResult] = []
    with tempfile.TemporaryDirectory(prefix="ess-dive-eval-") as tmp:
        for package in packages:
            results.append(run_dataset(package, Path(tmp), max_bytes))

    _print_summary(results)
    _write_report(results, out_label)
    _write_html(results, out_label)
    return results


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-")
    return cleaned or "ess-dive-eval"


def _print_summary(results: list[DatasetResult]) -> None:
    print("\n=== Summary ===")
    print(f"  {'dataset':<58}{'derivable':>10}{'complete':>10}{'fidelity':>10}")
    for r in results:
        if r.error or not r.score:
            print(f"  {r.name:<58}{(r.error or 'no score')[:38]:>30}")
            continue
        s = r.score
        print(f"  {r.name:<58}{s.derivable_score:>9.0%}{s.catalog_completeness:>10.0%}{len(s.fidelity_errors):>10}")


def _write_report(results: list[DatasetResult], label: str = "ess-dive-eval") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"{label}-{stamp}.json"
    payload = []
    for r in results:
        s = r.score
        payload.append({
            "name": r.name,
            "error": r.error,
            "scores": None if not s else {
                "derivable_metadata": s.derivable_score,
                "catalog_completeness": s.catalog_completeness,
                "schema_valid": s.schema_valid,
                "fidelity_errors": s.fidelity_errors,
                "unavailable_fields": s.unavailable_fields,
                "components": s.components,
                "gates_applied": s.gates_applied,
                "free_text_judgment": s.free_text_judgment,
            },
            "fields": None if not s else {
                fn: {"availability": fs.availability, "status": fs.status, "score": fs.score, "note": fs.note,
                     "fidelity_errors": fs.fidelity_errors, "proposed": fs.proposed, "standard": fs.standard}
                for fn, fs in s.fields.items()
            },
            "resources": [
                {
                    "name": rs.name,
                    "format": rs.resource.get("format"),
                    "url": rs.resource.get("url"),
                    "downloaded": rs.local_path is not None,
                    "note": rs.skipped_reason,
                    "gdal": gdal_capability(rs.resource),
                }
                for rs in r.staged
            ],
            "generated": r.generated,
        })
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nJSON report: {path}")
    return path


_HTML_HEAD = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>ESS-DIVE Round-Trip Evaluation</title><style>
 body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#1f2328;max-width:1150px}
 h1{font-size:1.5rem;margin-bottom:.2rem} h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #d0d7de;padding-bottom:.3rem}
 .muted{color:#656d76} table{border-collapse:collapse;width:100%;margin:.5rem 0}
 th,td{border:1px solid #d0d7de;padding:.4rem .6rem;text-align:left;vertical-align:top}
 th{background:#f6f8fa} td.num{text-align:center} code{background:#eff1f3;padding:.1rem .3rem;border-radius:4px;font-size:.85em}
 .score{font-size:1.6rem;font-weight:700} .scores{display:flex;gap:2.5rem;margin:.5rem 0}
 .pill{font-weight:700;font-size:.78rem;padding:.1rem .5rem;border-radius:10px}
 details summary{cursor:pointer;color:#0969da} pre{background:#f6f8fa;padding:.6rem;border-radius:6px;overflow:auto;font-size:.82em}
 .err{color:#cf222e} .gate{color:#9a6700}
</style></head><body>"""

_STATUS_CSS = {"match": "#1a7f37", "partial": "#9a6700", "differ": "#cf222e", "missing": "#cf222e", "n/a": "#656d76"}
_PULL_CSS = {"yes": "#1a7f37", "maybe": "#9a6700", "no": "#656d76"}

_EMPTYISH = {"", "null", "none", "n/a", "na"}


def _fmt_value(field_name: str, value: object, limit: int = 180) -> str:
    """Compact, HTML-escaped rendering of a field value for the Proposed/Standard columns."""
    import html

    if value is None or value == [] or value == {} or (isinstance(value, str) and value.strip().lower() in _EMPTYISH):
        return "<span class='muted'>&ndash;</span>"
    if field_name == "tags" and isinstance(value, list):
        names = [str(t.get("name")) for t in value if isinstance(t, dict) and t.get("name")]
        text = ", ".join(names)
    elif field_name == "extras" and isinstance(value, list):
        keys = [str(t.get("key")) for t in value if isinstance(t, dict) and t.get("key")]
        text = ", ".join(keys)
    elif field_name == "resources" and isinstance(value, list):
        text = f"{len(value)} resource(s)"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    truncated = text[:limit] + ("…" if len(text) > limit else "")
    return f"<span title='{html.escape(text[:600])}'>{html.escape(truncated)}</span>"


def render_html(results: list[DatasetResult]) -> str:
    import html
    from datetime import datetime

    out = [_HTML_HEAD, "<h1>ESS-DIVE Round-Trip Evaluation</h1>",
           f"<p class='muted'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · files-only inputs · "
           "resources injected from manifest</p>"]
    for r in results:
        out.append(f"<h2>{html.escape(r.name)}</h2>")
        if r.error or not r.score:
            out.append(f"<p class='err'>{html.escape(r.error or 'no score')}</p>")
            continue
        s = r.score
        out.append(
            "<div class='scores'>"
            f"<div><div class='score' style='color:#0969da'>{s.derivable_score:.0%}</div><div class='muted'>derivable metadata</div></div>"
            f"<div><div class='score'>{s.catalog_completeness:.0%}</div><div class='muted'>catalog completeness</div></div>"
            f"<div><div class='score' style='color:{'#cf222e' if s.fidelity_errors else '#1a7f37'}'>{len(s.fidelity_errors)}</div><div class='muted'>fidelity errors</div></div>"
            "</div>"
        )
        j = s.free_text_judgment or {}
        if j:
            out.append(f"<p class='muted'>judge: title {j.get('title_score')}/5 · notes {j.get('notes_score')}/5 · "
                       f"tags {j.get('tags_score')}/5 · faithful {j.get('faithfulness_pass')} — {html.escape(str(j.get('comment','')))}</p>")
        if s.gates_applied:
            out.append("<p class='gate'>gates: " + html.escape("; ".join(s.gates_applied)) + "</p>")
        if s.fidelity_errors:
            out.append("<p class='err'>fidelity errors:</p><ul class='err'>"
                       + "".join(f"<li>{html.escape(e)}</li>" for e in s.fidelity_errors) + "</ul>")

        # Fields table — Proposed (agent) vs Standard (gold)
        rows = []
        for fn, fs in s.fields.items():
            color = _STATUS_CSS.get(fs.status, "#1f2328")
            sc = "&ndash;" if fs.score is None else f"{fs.score:.2f}"
            avail = "<i class='muted'>excluded</i>" if fs.availability == "unavailable" else fs.availability
            rows.append(
                f"<tr><td><code>{fn}</code></td><td>{avail}</td>"
                f"<td style='color:{color};font-weight:600'>{fs.status}</td><td class='num'>{sc}</td>"
                f"<td>{_fmt_value(fn, fs.proposed)}</td>"
                f"<td class='muted'>{_fmt_value(fn, fs.standard)}</td>"
                f"<td class='muted' title='{html.escape(fs.note)}'>{html.escape(fs.note[:70])}</td></tr>"
            )
        out.append("<table><thead><tr><th>Field</th><th>Availability</th><th>Status</th><th>Score</th>"
                   "<th>Proposed (agent)</th><th>Standard (gold)</th><th>Note</th></tr></thead><tbody>"
                   + "".join(rows) + "</tbody></table>")

        # GDAL pullability table
        grows = []
        for st in r.staged:
            cap = gdal_capability(st.resource)
            color = _PULL_CSS.get(cap["pullable"], "#1f2328")
            grows.append(f"<tr><td>{html.escape(st.name)}</td><td>{html.escape(str(st.resource.get('format','')))}</td>"
                         f"<td style='color:{color};font-weight:600'>{cap['pullable']}</td>"
                         f"<td>{html.escape(cap['driver'])}</td><td class='muted'>{html.escape(cap['note'])}</td></tr>")
        out.append("<details><summary>GDAL pullability ("
                   + f"{sum(1 for st in r.staged if gdal_capability(st.resource)['pullable']=='yes')} pullable)</summary>"
                   "<table><thead><tr><th>Resource</th><th>Format</th><th>Pullable</th><th>Driver</th><th>Note</th></tr></thead><tbody>"
                   + "".join(grows) + "</tbody></table></details>")

        # Generated package
        out.append("<details><summary>generated CKAN package</summary><pre>"
                   + html.escape(json.dumps(r.generated, indent=2, ensure_ascii=False)) + "</pre></details>")
    out.append("</body></html>")
    return "\n".join(out)


def _write_html(results: list[DatasetResult], label: str = "ess-dive-eval") -> Path:
    from datetime import datetime

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"{label}-{stamp}.html"
    path.write_text(render_html(results), encoding="utf-8")
    print(f"HTML report: {path}")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ESS-DIVE round-trip eval: files -> agent CKAN metadata -> compare to gold.",
        epilog='Custom fixture JSON shape: {"datasets": [{"ckan_package": {"name": ..., "title": ..., '
        '"notes": ..., "tags": [...], "resources": [{"name":...,"format":...,"url":...}]}}]}',
    )
    parser.add_argument("--fixture", "--input", dest="fixture",
                        help="Path to a datasets JSON to evaluate. Defaults to the bundled ESS-DIVE fixture.")
    parser.add_argument("--label", help="Prefix for the output JSON/HTML filenames. Defaults to the fixture filename.")
    parser.add_argument("--dataset", help="Only run this ckan_package name.")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Per-file download cap (bytes).")
    parser.add_argument("--limit", type=int, help="Only run the first N datasets.")
    args = parser.parse_args()

    run(dataset_filter=args.dataset, max_bytes=args.max_bytes, limit=args.limit,
        fixture=args.fixture, label=args.label)


if __name__ == "__main__":
    main()
