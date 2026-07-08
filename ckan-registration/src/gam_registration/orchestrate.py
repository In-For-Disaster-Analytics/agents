"""End-to-end GAM registration orchestration (Capabilities A/B/D integration).

This module wires together the already-implemented Capability A/B/D modules into
a single per-model registration pipeline that runs on Lonestar6 (or any host with
the Corral GAM filesystem locally mounted).

Orchestration steps per model (see Data Flow in the design spec):
  1. Build resource plan from local files (list_resource_files + build_resource_plan).
  2. MINT-annotate the resource plan (annotate_resource_plan_with_mint_standard_variables).
  3. TWDB enrichment:
     a. Fetch landing page (fetch_source_metadata).
     b. Discover report URL (twdb_enrich.discover_report_url, with manifest override).
     c. PDF map-reduce via pdf_extract.run_pdf_map_reduce (if report_pdf_url available).
     d. No-PDF fallback: use landing-page-only metadata (existing propose_ckan_dataset_metadata_with_llm path).
  4. Run three-persona loop (persona_loop.run_persona_metadata_loop).
  5. Map to SUBSIDE schema (subside_mapping.map_to_subside_dataset).
  6. Attach TWDB link resources (twdb_enrich.build_link_resources).
  7. Dry-run diff (surfaces link resources by name + URL alongside package body).
  8. Apply (GATED — only called with approval="REGISTER").

Backward-compatible no-PDF path:
  If `report_pdf_url` is None (no PDF discovered and no manifest override),
  the PDF map-reduce step is skipped. `propose_ckan_dataset_metadata_with_llm`
  is used to produce the consolidated_inputs dict from the landing-page excerpt
  only — the same single-LLM-call behavior as the current (pre-Capability-B) code.
  The persona loop and SUBSIDE mapping then proceed normally with this dict.

Dry-run link-resource surfacing:
  The returned dry-run dict includes a ``link_resource_plan`` key with the list
  of url-type link resource dicts (name + URL + format). This is surfaced in the
  notebook dry-run review cell so users can verify landing-page and PDF link
  resources before proceeding to apply.

Apply gate:
  `run_registration` accepts an ``approval`` string. Apply is only called when
  ``approval == "REGISTER"``. All CKAN writes (package_create/patch,
  resource_create/update, create_link_resources) remain inside the approval gate.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import of optional modules so unit tests can patch them without the
# full dependency chain being required at import time.
# ---------------------------------------------------------------------------
from . import utils as _u
from . import twdb_enrich as _twdb
from . import subside_mapping as _sm
from . import persona_loop as _pl
from . import resource_review as _rr
from . import tapis_links as _tl

# pdf_extract is imported lazily inside _run_pdf_enrichment to keep the import
# isolated (pymupdf is optional; an ImportError is caught and the no-PDF path
# is used instead).


# ---------------------------------------------------------------------------
# GAM defaults applied at mapping time
# ---------------------------------------------------------------------------
_GAM_DEFAULTS: dict[str, Any] = {
    "collection_method": "Model Output",
    "categories": ["Groundwater"],
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    """Result of a single-model orchestration run.

    Attributes
    ----------
    model_id:
        The dataset ``name`` slug from the manifest.
    ok:
        True if the pipeline completed without a fatal error.
    dry_run_summary:
        Dict with keys: ``package_body``, ``changes``, ``link_resource_plan``,
        ``persona_loop_converged``, ``persona_loop_rounds``,
        ``outstanding_questions``.  Always populated (even on no-apply runs).
    apply_result:
        Dict returned by CKAN on apply, or None if apply was not performed.
    persona_loop_result:
        The raw :class:`persona_loop.PersonaLoopResult` (or None on failure).
    error:
        Error message string if ok=False, otherwise empty string.
    """
    model_id: str
    ok: bool = True
    dry_run_summary: dict[str, Any] = field(default_factory=dict)
    apply_result: dict[str, Any] | None = None
    persona_loop_result: Any = None  # PersonaLoopResult
    error: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sleep_between_calls() -> None:
    """Honour LLM_CALL_DELAY_SECONDS between LLM/API calls."""
    delay = float(os.environ.get("LLM_CALL_DELAY_SECONDS", "4"))
    if delay > 0:
        time.sleep(delay)


def _is_geodatabase_path(path: Path) -> bool:
    """True if *path* lives inside an ESRI File Geodatabase (a ``*.gdb``
    directory) or a ``Geodatabase`` folder.  These are the spatial/CRS source
    (already mined for bbox + coordinate_system) and are NOT uploaded as
    CKAN resources.
    """
    for part in path.parts:
        pl = part.lower()
        if pl.endswith(".gdb") or pl == "geodatabase":
            return True
    return False


def _run_pdf_enrichment(
    report_pdf_url: str,
    landing_page_excerpt: str,
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None,
) -> dict[str, Any]:
    """Run PDF map-reduce and return the consolidated metadata dict.

    Returns an empty dict if pdf_extract is unavailable (pymupdf not installed)
    or if PDF extraction yields no text.
    """
    try:
        from . import pdf_extract as _pdf  # noqa: F401 (lazy import)
    except ImportError:
        logger.warning(
            "orchestrate: pdf_extract unavailable (pymupdf not installed); "
            "falling back to landing-page-only metadata."
        )
        return {}

    try:
        temp_path = _pdf.fetch_pdf_to_temp(report_pdf_url)
    except Exception as exc:
        logger.warning(
            "orchestrate: failed to fetch report PDF %s: %s; "
            "using landing-page-only metadata.",
            report_pdf_url,
            exc,
        )
        return {}

    try:
        return _pdf.run_pdf_map_reduce(
            temp_path,
            landing_page_excerpt,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
    except Exception as exc:
        logger.warning(
            "orchestrate: PDF map-reduce failed for %s: %s; "
            "using landing-page-only metadata.",
            report_pdf_url,
            exc,
        )
        return {}
    finally:
        # Best-effort temp file cleanup.
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def _run_local_pdf_enrichment(
    local_path: Path,
    landing_page_excerpt: str,
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None,
) -> dict[str, Any]:
    """Run PDF map-reduce on a LOCAL file (no download, no temp cleanup).

    The file is owned by the caller's filesystem and must NOT be deleted here.
    Returns an empty dict if pdf_extract is unavailable or map-reduce fails.
    """
    try:
        from . import pdf_extract as _pdf  # noqa: F401 (lazy import)
    except ImportError:
        logger.warning(
            "orchestrate: pdf_extract unavailable (pymupdf not installed); "
            "falling back to landing-page-only metadata."
        )
        return {}

    logger.info("orchestrate: using local report PDF: %s", local_path)
    try:
        return _pdf.run_pdf_map_reduce(
            local_path,
            landing_page_excerpt,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
    except Exception as exc:
        logger.warning(
            "orchestrate: PDF map-reduce failed for local file %s: %s; "
            "using landing-page-only metadata.",
            local_path,
            exc,
        )
        return {}
    # No cleanup — this is a real file, not a temp.


def _build_consolidated_inputs_no_pdf(
    model_record: dict[str, Any],
    resource_plan: list[dict[str, Any]],
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None,
) -> dict[str, Any]:
    """Backward-compatible single-LLM-call path when no PDF is available.

    Calls propose_ckan_dataset_metadata_with_llm with the landing-page URL
    and returns its output dict as consolidated_inputs.
    """
    return _u.propose_ckan_dataset_metadata_with_llm(
        resource_plan,
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        source_metadata_url=model_record.get("twdb_page_url") or "",
        preferred_dataset_name=model_record.get("package_id") or "",
        preferred_dataset_title=model_record.get("title") or "",
        preferred_dataset_url=model_record.get("twdb_page_url") or "",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_registration(
    model_record: dict[str, Any],
    *,
    ckan_url: str,
    auth_header: str | None = None,
    owner_org: str | None = None,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None = None,
    approval: str = "",
    max_rounds: int = 3,
    runs_dir: Path | None = None,
    subside_schema_fields: list[dict[str, Any]] | None = None,
    mint_standard_variable_names: list[str] | None = None,
    org_defaults: dict[str, Any] | None = None,
    review_resources: bool = False,
    register_by_reference: bool = False,
) -> RegistrationResult:
    """Run the full GAM registration pipeline for a single model record.

    Parameters
    ----------
    model_record:
        One entry from the manifest JSON.  Required keys: ``package_folder``,
        ``package_id``.  Optional keys: ``twdb_page_url``, ``report_url``,
        ``boundary_bbox_geojson``, ``dataset_spatial``.
    ckan_url:
        Base CKAN instance URL.
    auth_header:
        Authorization header value (Bearer JWT or API key).
    owner_org:
        CKAN organization slug for the dataset.
    llm_model / llm_api_key / llm_base_url:
        LLM endpoint configuration.
    approval:
        Pass ``"REGISTER"`` to execute the apply step.  Any other value
        (including empty string) performs dry-run only.
    max_rounds:
        Maximum rounds for the persona loop.
    runs_dir:
        Directory for persona-loop audit transcripts.  Defaults to
        ``Path("runs/")`` in the current working directory.
    subside_schema_fields:
        Optional list of field dicts describing the subside_dataset schema
        (passed to persona loop for author context).
    mint_standard_variable_names:
        Optional list of MINT standard variable names to annotate resources with.
    org_defaults:
        Optional dict of authoritative organizational/config values to seed into the
        author persona and backfill into the final package body.  Expected keys (all
        optional): ``license_id``, ``author``, ``author_email``, ``maintainer``,
        ``maintainer_email``, ``owner_org``, ``data_contact_email``.  Empty/None
        values are ignored.  ``owner_org`` here acts as a fallback for the explicit
        ``owner_org`` parameter (which takes precedence).  All other callers are
        unaffected when ``org_defaults=None`` (the default).
    review_resources:
        If True and an LLM API key is available, call
        ``resource_review.review_resource_descriptions`` after the resource plan
        is built and MINT-annotated to improve descriptions via a batched LLM
        call before mapping/upload.  Default False keeps existing behaviour.
    register_by_reference:
        If True, skip byte-uploading file resources and instead mint a Tapis
        postit per file and register it as a url-type CKAN resource.  Requires
        ``TAPIS_SYSTEM_ID`` and ``TAPIS_SYSTEM_ROOTDIR`` env vars, and
        ``auth_header`` must be a Bearer JWT (i.e. ``CKAN_AUTH_MODE=tapis_password``).
        ``TAPIS_FILES_BASE_URL`` defaults to the tenant base derived from
        ``DEFAULT_TAPIS_URL`` (``https://portals.tapis.io``).
        Default False keeps the existing byte-upload path unchanged.

    Returns
    -------
    RegistrationResult
        Always returns — never raises.  Failures are captured in the
        ``error`` field; the outer 23-model loop continues uninterrupted.
    """
    model_id = _u.clean_text(model_record.get("package_id") or model_record.get("name") or "")
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        result = _run_registration_inner(
            model_record,
            model_id=model_id,
            run_timestamp=run_timestamp,
            ckan_url=ckan_url,
            auth_header=auth_header,
            owner_org=owner_org,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            approval=approval,
            max_rounds=max_rounds,
            runs_dir=runs_dir,
            subside_schema_fields=subside_schema_fields,
            mint_standard_variable_names=mint_standard_variable_names,
            org_defaults=org_defaults,
            review_resources=review_resources,
            register_by_reference=register_by_reference,
        )
        return result
    except Exception as exc:
        logger.exception(
            "orchestrate: unhandled error for model %s: %s", model_id, exc
        )
        return RegistrationResult(
            model_id=model_id,
            ok=False,
            error=f"Unhandled error: {exc}",
        )


def _run_registration_inner(
    model_record: dict[str, Any],
    *,
    model_id: str,
    run_timestamp: str,
    ckan_url: str,
    auth_header: str | None,
    owner_org: str | None,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None,
    approval: str,
    max_rounds: int,
    runs_dir: Path | None,
    subside_schema_fields: list[dict[str, Any]] | None,
    mint_standard_variable_names: list[str] | None,
    org_defaults: dict[str, Any] | None = None,
    review_resources: bool = False,
    register_by_reference: bool = False,
) -> RegistrationResult:
    """Inner orchestration; called from run_registration under try/except."""

    package_folder = _u.clean_text(model_record.get("package_folder") or "")
    if not package_folder:
        raise ValueError(f"model_record for {model_id} has no package_folder.")
    package_path = Path(package_folder)

    # ------------------------------------------------------------------
    # Step 1: Build resource plan from local files.
    # ------------------------------------------------------------------
    logger.info("orchestrate [%s]: listing resource files in %s", model_id, package_path)
    files = _u.list_resource_files(package_path)
    # The geodatabase is the spatial/CRS source (already mined for bbox + CRS),
    # not an upload target — exclude .gdb contents and any 'Geodatabase' folder.
    _n_before = len(files)
    files = [f for f in files if not _is_geodatabase_path(f)]
    _n_gdb = _n_before - len(files)
    if _n_gdb:
        logger.info("orchestrate [%s]: excluded %d geodatabase file(s) from upload", model_id, _n_gdb)
    twdb_page_url = _u.clean_text(model_record.get("twdb_page_url") or "")
    resource_plan = _u.build_resource_plan(files, package_path, twdb_page_url or package_folder)
    logger.info("orchestrate [%s]: %d files in resource plan", model_id, len(resource_plan))

    # ------------------------------------------------------------------
    # Step 1b: Build compact file inventory for the Domain Expert author.
    # ------------------------------------------------------------------
    _FILENAMES_CAP = 200
    _filenames = [f.name for f in files]
    _truncated = len(_filenames) > _FILENAMES_CAP
    file_inventory: dict[str, Any] = {
        "file_count": len(files),
        "extension_counts": _u.summarize_extensions(files),
        "filenames": _filenames[:_FILENAMES_CAP],
    }
    if _truncated:
        file_inventory["filenames_truncated"] = True
    logger.info(
        "orchestrate [%s]: file_inventory file_count=%d truncated=%s",
        model_id,
        file_inventory["file_count"],
        _truncated,
    )

    # ------------------------------------------------------------------
    # Step 2: MINT-annotate the resource plan.
    # ------------------------------------------------------------------
    if mint_standard_variable_names:
        resource_plan = _u.annotate_resource_plan_with_mint_standard_variables(
            resource_plan,
            standard_variable_names=mint_standard_variable_names,
        )
        logger.debug(
            "orchestrate [%s]: annotated resource plan with %d MINT variables",
            model_id,
            len(mint_standard_variable_names),
        )

    # ------------------------------------------------------------------
    # Step 2b: (Optional) Batched LLM resource-description review.
    # ------------------------------------------------------------------
    if review_resources and _u.clean_text(llm_api_key):
        _dataset_context = (
            _u.clean_text(model_record.get("title") or model_record.get("package_id") or model_id)
        )
        logger.info(
            "orchestrate [%s]: running batched resource-description review "
            "(resource_plan size=%d)",
            model_id,
            len(resource_plan),
        )
        _n_updated = _rr.review_resource_descriptions(
            resource_plan,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            dataset_context=_dataset_context,
        )
        logger.info(
            "orchestrate [%s]: resource-description review updated %d description(s)",
            model_id,
            _n_updated,
        )

    # ------------------------------------------------------------------
    # Step 3a: Fetch TWDB landing page.
    # ------------------------------------------------------------------
    landing_page_excerpt = ""
    if twdb_page_url:
        try:
            source_meta = _u.fetch_source_metadata(twdb_page_url)
            landing_page_excerpt = source_meta.get("excerpt") or ""
            logger.debug(
                "orchestrate [%s]: fetched landing page excerpt (%d chars)",
                model_id,
                len(landing_page_excerpt),
            )
        except Exception as exc:
            logger.warning(
                "orchestrate [%s]: failed to fetch landing page %s: %s",
                model_id,
                twdb_page_url,
                exc,
            )

    # ------------------------------------------------------------------
    # Step 3b: Find local report PDF OR discover remote URL (LOCAL-FIRST).
    # ------------------------------------------------------------------
    manifest_report_url = _u.clean_text(model_record.get("report_url") or "")
    report_pdf_url: str | None = None
    _report_from_local: bool = False  # True when the report was sourced from disk

    try:
        from . import pdf_extract as _pdf_finder
        local_report_pdf = _pdf_finder.find_local_report_pdf(package_path)
    except ImportError:
        local_report_pdf = None

    if local_report_pdf is not None:
        logger.info(
            "orchestrate [%s]: found local report PDF: %s", model_id, local_report_pdf
        )
    else:
        # No local PDF — attempt URL discovery.
        if twdb_page_url or manifest_report_url:
            try:
                report_pdf_url = _twdb.discover_report_url(
                    twdb_page_url or "",
                    report_url_override=manifest_report_url or None,
                )
                if report_pdf_url:
                    logger.info(
                        "orchestrate [%s]: report PDF URL: %s", model_id, report_pdf_url
                    )
                else:
                    logger.info(
                        "orchestrate [%s]: no report PDF URL found", model_id
                    )
            except Exception as exc:
                logger.warning(
                    "orchestrate [%s]: report URL discovery failed: %s", model_id, exc
                )

    # ------------------------------------------------------------------
    # Step 3c/d: PDF map-reduce (local or URL) OR no-PDF fallback.
    # ------------------------------------------------------------------
    if local_report_pdf is not None:
        # Local-first path: map-reduce directly on the on-disk file.
        logger.info("orchestrate [%s]: running PDF map-reduce on local file", model_id)
        _sleep_between_calls()
        _report_from_local = True
        consolidated_inputs = _run_local_pdf_enrichment(
            local_report_pdf,
            landing_page_excerpt,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
        if not consolidated_inputs:
            # Local PDF enrichment returned empty — fall back to landing-page-only.
            logger.info(
                "orchestrate [%s]: local PDF enrichment returned empty; falling back to "
                "landing-page-only proposal",
                model_id,
            )
            _sleep_between_calls()
            consolidated_inputs = _build_consolidated_inputs_no_pdf(
                model_record,
                resource_plan,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )
    elif report_pdf_url:
        # URL fallback path: download and map-reduce.
        logger.info("orchestrate [%s]: running PDF map-reduce (URL fallback)", model_id)
        _sleep_between_calls()
        consolidated_inputs = _run_pdf_enrichment(
            report_pdf_url,
            landing_page_excerpt,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )
        if not consolidated_inputs:
            # PDF enrichment returned empty — fall back to landing-page-only.
            logger.info(
                "orchestrate [%s]: PDF enrichment returned empty; falling back to "
                "landing-page-only proposal",
                model_id,
            )
            _sleep_between_calls()
            consolidated_inputs = _build_consolidated_inputs_no_pdf(
                model_record,
                resource_plan,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )
    else:
        # No-PDF path (backward-compatible): single LLM call from landing page.
        logger.info(
            "orchestrate [%s]: no PDF available; using landing-page-only proposal",
            model_id,
        )
        _sleep_between_calls()
        consolidated_inputs = _build_consolidated_inputs_no_pdf(
            model_record,
            resource_plan,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
        )

    # ------------------------------------------------------------------
    # Step 4: Three-persona loop.
    # ------------------------------------------------------------------
    bbox_geojson = _u.clean_text(
        model_record.get("boundary_bbox_geojson") or model_record.get("dataset_spatial") or ""
    )

    # Build organizational_metadata from org_defaults (drop empty/None values).
    # The explicit owner_org parameter takes precedence over org_defaults.
    _ORG_KEYS = (
        "license_id", "author", "author_email", "maintainer", "maintainer_email",
        "owner_org", "data_contact_email",
    )
    organizational_metadata: dict[str, Any] = {}
    if org_defaults:
        for _k in _ORG_KEYS:
            _v = org_defaults.get(_k)
            if _v is not None and _v != "":
                organizational_metadata[_k] = _v
    # The explicit owner_org param always wins over org_defaults.
    if owner_org:
        organizational_metadata["owner_org"] = owner_org
    # Seed model-specific authoritative values so the author populates them
    # in-loop and the FAIR evaluator does not re-ask (url, CRS from the gdb).
    if twdb_page_url:
        organizational_metadata["url"] = twdb_page_url
    _cs = _u.clean_text(model_record.get("coordinate_system") or "")
    if _cs:
        organizational_metadata["coordinate_system"] = _cs

    logger.info("orchestrate [%s]: running persona loop (max_rounds=%d)", model_id, max_rounds)
    persona_result = _pl.run_persona_metadata_loop(
        consolidated_inputs,
        resource_plan=resource_plan,
        bbox_geojson=bbox_geojson or None,
        subside_schema_fields=subside_schema_fields,
        gam_defaults=_GAM_DEFAULTS,
        max_rounds=max_rounds,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        model_id=model_id,
        run_timestamp=run_timestamp,
        runs_dir=runs_dir,
        file_inventory=file_inventory,
        organizational_metadata=organizational_metadata or None,
    )
    logger.info(
        "orchestrate [%s]: persona loop done — converged=%s rounds=%d",
        model_id,
        persona_result.converged,
        persona_result.rounds,
    )

    # ------------------------------------------------------------------
    # Step 5: Map to SUBSIDE schema.
    # ------------------------------------------------------------------
    proposed_metadata = dict(persona_result.proposed_metadata or consolidated_inputs)

    # Enhancement A: if proposed_metadata has no url, fall back to twdb_page_url.
    if not proposed_metadata.get("url") and not proposed_metadata.get("dataset_url"):
        fallback_url = _u.clean_text(model_record.get("twdb_page_url") or "")
        if fallback_url:
            proposed_metadata["url"] = fallback_url
            logger.info(
                "orchestrate [%s]: backfilling dataset url from twdb_page_url: %s",
                model_id,
                fallback_url,
            )

    # Enhancement B: if model_record carries coordinate_system (from gdb discovery),
    # inject it into proposed_metadata so it reaches subside_mapping extras.
    model_crs = _u.clean_text(model_record.get("coordinate_system") or "")
    if model_crs and not proposed_metadata.get("coordinate_system"):
        proposed_metadata["coordinate_system"] = model_crs
        logger.info(
            "orchestrate [%s]: injecting coordinate_system from discovery: %s",
            model_id,
            model_crs,
        )

    # Enhancement C: backfill org_defaults fields that the author left null/empty.
    # This ensures config-supplied values (license, maintainer, etc.) appear in the
    # final package even when the author could not derive them from document sources.
    _BACKFILL_KEYS = (
        "license_id", "author", "author_email", "maintainer", "maintainer_email",
        "owner_org", "data_contact_email",
    )
    if org_defaults:
        for _k in _BACKFILL_KEYS:
            _default_v = org_defaults.get(_k)
            if _default_v is not None and _default_v != "":
                if not proposed_metadata.get(_k):
                    proposed_metadata[_k] = _default_v
                    logger.info(
                        "orchestrate [%s]: backfilling %s from org_defaults", model_id, _k
                    )
    # The explicit owner_org param always wins.
    if owner_org:
        proposed_metadata["owner_org"] = owner_org

    package_body = _sm.map_to_subside_dataset(
        proposed_metadata,
        spatial=bbox_geojson or None,
        owner_org=owner_org or None,
        gam_defaults=_GAM_DEFAULTS,
    )
    logger.debug("orchestrate [%s]: package_body type=%s", model_id, package_body.get("type"))

    # ------------------------------------------------------------------
    # Step 6: Build TWDB link resources.
    # ------------------------------------------------------------------
    # When the report was sourced from a LOCAL file it will be uploaded as a
    # normal file resource by the recursive resource_plan walk.  Do NOT add a
    # web link for it.  Pass the URL only when the report came from a remote URL.
    _link_report_url = None if _report_from_local else report_pdf_url
    link_resources = _twdb.build_link_resources(
        twdb_page_url or None,
        _link_report_url,
    )
    logger.debug(
        "orchestrate [%s]: %d link resources built", model_id, len(link_resources)
    )

    # ------------------------------------------------------------------
    # Step 7: Dry-run diff — surfaces link resources by name + URL.
    # ------------------------------------------------------------------
    outstanding_questions: list[str] = []
    for loop_round in (persona_result.transcript or []):
        for evaluator in (loop_round.fair_evaluator, loop_round.usability_evaluator):
            if evaluator and evaluator.verdict == "revise":
                outstanding_questions.extend(evaluator.questions or [])

    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped_questions: list[str] = []
    for q in outstanding_questions:
        if q not in seen:
            seen.add(q)
            deduped_questions.append(q)

    link_resource_plan_summary = [
        {"name": lr.get("name"), "url": lr.get("url"), "format": lr.get("format")}
        for lr in link_resources
    ]

    dry_run_summary: dict[str, Any] = {
        "package_body": package_body,
        "extras": package_body.get("extras", []),
        "link_resource_plan": link_resource_plan_summary,
        "persona_loop_converged": persona_result.converged,
        "persona_loop_rounds": persona_result.rounds,
        "persona_loop_stop_reason": persona_result.stop_reason,
        "outstanding_questions": deduped_questions,
    }

    # If persona loop did not converge, surface questions prominently.
    if not persona_result.converged and deduped_questions:
        logger.warning(
            "orchestrate [%s]: persona loop did not converge. "
            "%d outstanding question(s) require human resolution before apply.",
            model_id,
            len(deduped_questions),
        )

    # ------------------------------------------------------------------
    # Step 8: Apply — GATED on approval="REGISTER".
    # ------------------------------------------------------------------
    apply_result: dict[str, Any] | None = None
    if (_u.clean_text(approval) or "").upper() == "REGISTER":
        logger.info(
            "orchestrate [%s]: REGISTER approval received — proceeding with apply.",
            model_id,
        )
        dataset_name = _u.clean_text(package_body.get("name") or model_id)
        # extras: package_body["extras"] is a list of {"key":..., "value":...}
        # passed via dataset_extras so create_or_update sets create_payload["extras"].
        # mint_standard_variables is a live dataset column — passed via extra_fields
        # since create_or_update_ckan_dataset has no dedicated parameter for it.
        pkg_extras: list[dict[str, str]] = package_body.get("extras") or []
        pkg_extra_fields: dict[str, Any] = {}
        if package_body.get("mint_standard_variables"):
            pkg_extra_fields["mint_standard_variables"] = package_body["mint_standard_variables"]

        dataset_after = _u.create_or_update_ckan_dataset(
            ckan_url,
            dataset_name=dataset_name,
            dataset_title=package_body.get("title") or dataset_name,
            dataset_notes=package_body.get("notes") or "",
            dataset_tags=package_body.get("tags") or [],
            auth_header=auth_header,
            owner_org=package_body.get("owner_org"),
            dataset_author=package_body.get("author"),
            dataset_author_email=package_body.get("author_email"),
            dataset_maintainer=package_body.get("maintainer"),
            dataset_maintainer_email=package_body.get("maintainer_email"),
            dataset_license_id=package_body.get("license_id"),
            dataset_url=package_body.get("url"),
            dataset_version=package_body.get("version"),
            dataset_type="subside_dataset",
            temporal_coverage_start=package_body.get("temporal_coverage_start"),
            temporal_coverage_end=package_body.get("temporal_coverage_end"),
            dataset_extras=pkg_extras or None,
            extra_fields=pkg_extra_fields or None,
        )

        if register_by_reference:
            # ------------------------------------------------------------------
            # Register-by-reference path: mint Tapis postits, register as
            # url-type link resources.  No byte upload.
            # ------------------------------------------------------------------
            _tapis_system_id = _u.clean_text(os.environ.get("TAPIS_SYSTEM_ID", ""))
            _tapis_rootdir = _u.clean_text(os.environ.get("TAPIS_SYSTEM_ROOTDIR", ""))
            if not _tapis_system_id or not _tapis_rootdir:
                raise RuntimeError(
                    "orchestrate [%s]: register_by_reference=True but "
                    "TAPIS_SYSTEM_ID and/or TAPIS_SYSTEM_ROOTDIR env vars are not set. "
                    "Set both before using register-by-reference mode." % model_id
                )

            # Derive the Tapis JWT from auth_header (must be Bearer <jwt>).
            _auth = _u.clean_text(auth_header)
            if not _auth.startswith("Bearer "):
                raise RuntimeError(
                    "orchestrate [%s]: register_by_reference=True requires a Bearer JWT "
                    "auth_header (set CKAN_AUTH_MODE=tapis_password so auth_header is "
                    "'Bearer <jwt>').  Got: %r" % (model_id, _auth[:40] if _auth else "")
                )
            _jwt = _auth[len("Bearer "):]

            # Determine Tapis Files base URL.
            _default_base = "https://portals.tapis.io"
            _tapis_files_base = _u.clean_text(
                os.environ.get("TAPIS_FILES_BASE_URL", "")
            ) or _default_base

            _postit_valid_seconds = int(
                os.environ.get("POSTIT_VALID_SECONDS", "3153600000")
            )
            _postit_allowed_uses = int(
                os.environ.get("POSTIT_ALLOWED_USES", "-1")
            )

            logger.info(
                "orchestrate [%s]: register_by_reference — minting postits for %d file(s)",
                model_id, len(resource_plan),
            )
            tapis_link_resources = _tl.build_tapis_link_resources(
                resource_plan,
                system_id=_tapis_system_id,
                system_root_dir=_tapis_rootdir,
                base_url=_tapis_files_base,
                jwt=_jwt,
                allowed_uses=_postit_allowed_uses,
                valid_seconds=_postit_valid_seconds,
            )
            logger.info(
                "orchestrate [%s]: minted %d postit(s) (%d skipped)",
                model_id, len(tapis_link_resources),
                len(resource_plan) - len(tapis_link_resources),
            )

            # Combine Tapis link resources with TWDB landing-page link resources.
            all_link_resources = tapis_link_resources + link_resources

            link_results = _u.create_link_resources(
                ckan_url,
                dataset_after,
                all_link_resources,
                auth_header,
            )

            dataset_url_ckan = f"{ckan_url.rstrip('/')}/dataset/{dataset_after.get('name') or dataset_name}"
            apply_result = {
                "dataset_name": dataset_after.get("name") or dataset_name,
                "dataset_url": dataset_url_ckan,
                "link_resources_created": len(link_results),
                "register_by_reference": True,
                "postits_minted": len(tapis_link_resources),
            }
            logger.info(
                "orchestrate [%s]: apply (by-reference) complete — %s",
                model_id, dataset_url_ckan,
            )

        else:
            # ------------------------------------------------------------------
            # Default byte-upload path (unchanged).
            # ------------------------------------------------------------------
            # Upsert file resources.
            _u.upsert_resources(
                ckan_url,
                dataset_after,
                resource_plan,
                auth_header,
            )

            # Create/update link resources (url-type — no byte upload).
            link_results = _u.create_link_resources(
                ckan_url,
                dataset_after,
                link_resources,
                auth_header,
            )

            dataset_url_ckan = f"{ckan_url.rstrip('/')}/dataset/{dataset_after.get('name') or dataset_name}"
            apply_result = {
                "dataset_name": dataset_after.get("name") or dataset_name,
                "dataset_url": dataset_url_ckan,
                "link_resources_created": len(link_results),
            }
            logger.info(
                "orchestrate [%s]: apply complete — %s", model_id, dataset_url_ckan
            )
    else:
        logger.info(
            "orchestrate [%s]: dry-run only (approval=%r). "
            "Pass approval='REGISTER' to apply.",
            model_id,
            approval,
        )

    return RegistrationResult(
        model_id=model_id,
        ok=True,
        dry_run_summary=dry_run_summary,
        apply_result=apply_result,
        persona_loop_result=persona_result,
    )


# ---------------------------------------------------------------------------
# Convenience: run the full 23-model registration loop from the manifest.
# ---------------------------------------------------------------------------

def run_manifest_registration(
    manifest: list[dict[str, Any]],
    *,
    ckan_url: str,
    auth_header: str | None = None,
    owner_org: str | None = None,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None = None,
    approval: str = "",
    max_rounds: int = 3,
    runs_dir: Path | None = None,
    subside_schema_fields: list[dict[str, Any]] | None = None,
    mint_standard_variable_names: list[str] | None = None,
    org_defaults: dict[str, Any] | None = None,
    review_resources: bool = False,
    register_by_reference: bool = False,
) -> list[RegistrationResult]:
    """Run run_registration for every model in *manifest*.

    Errors on individual models are captured in RegistrationResult.error;
    the loop continues to the next model on failure (per spec LLM-failure-handling
    design decision).

    Parameters mirror run_registration — see that function's docstring.

    Returns
    -------
    list[RegistrationResult]
        One result per manifest entry, in manifest order.
    """
    Path("runs/").mkdir(parents=True, exist_ok=True)

    results: list[RegistrationResult] = []
    for i, model_record in enumerate(manifest):
        model_id = _u.clean_text(
            model_record.get("package_id") or model_record.get("name") or f"model-{i}"
        )
        logger.info(
            "orchestrate: processing model %d/%d: %s",
            i + 1,
            len(manifest),
            model_id,
        )
        result = run_registration(
            model_record,
            ckan_url=ckan_url,
            auth_header=auth_header,
            owner_org=owner_org,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            approval=approval,
            max_rounds=max_rounds,
            runs_dir=runs_dir,
            subside_schema_fields=subside_schema_fields,
            mint_standard_variable_names=mint_standard_variable_names,
            org_defaults=org_defaults,
            review_resources=review_resources,
            register_by_reference=register_by_reference,
        )
        results.append(result)
        if not result.ok:
            logger.error(
                "orchestrate: model %s FAILED: %s", model_id, result.error
            )
        else:
            logger.info("orchestrate: model %s OK", model_id)

    ok_count = sum(1 for r in results if r.ok)
    logger.info(
        "orchestrate: manifest loop complete — %d/%d models OK",
        ok_count,
        len(results),
    )
    return results
