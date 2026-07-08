"""Three-persona iterative metadata authoring and evaluation loop (Capability D).

This module is an independently testable and independently deletable prototype.
It must NOT be imported from utils.py or called from propose_ckan_dataset_metadata_with_llm.
The notebook/orchestrator calls this module explicitly after the map-reduce consolidation step.

LLM helper reuse:
    _chat_completion_content and _parse_llm_json are imported from utils.
    If utils is not importable, a thin fallback is defined (see import block).

Environment variables:
    LLM_CALL_DELAY_SECONDS  — sleep between successive LLM calls (default: 4.0)
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# LLM helper import — reuse from utils; do NOT reimplement the HTTP call.
# ---------------------------------------------------------------------------
try:
    from .utils import _chat_completion_content, _parse_llm_json  # type: ignore[import]
except ImportError:  # pragma: no cover — only hit if utils.py is unavailable
    # Thin stub so the module is still importable in test environments that
    # mock this at the call site.
    def _chat_completion_content(**kwargs: Any) -> str:  # type: ignore[misc]
        raise RuntimeError(
            "utils._chat_completion_content is not importable; "
            "ensure utils.py is on sys.path or mock this function in tests."
        )

    def _parse_llm_json(content: str) -> dict[str, Any]:  # type: ignore[misc]
        import re as _re, json as _json
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = _re.sub(r"\s*```$", "", cleaned)
        try:
            payload = _json.loads(cleaned)
            if isinstance(payload, dict):
                return payload
        except _json.JSONDecodeError:
            pass
        match = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
        if match:
            try:
                payload = _json.loads(match.group(0))
                if isinstance(payload, dict):
                    return payload
            except _json.JSONDecodeError:
                pass
        return {}


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Throttle delay (configurable via env; default 1 second).
# ---------------------------------------------------------------------------
LLM_CALL_DELAY_SECONDS: float = float(os.environ.get("LLM_CALL_DELAY_SECONDS", "4.0"))

# ---------------------------------------------------------------------------
# Persona prompt templates
# ---------------------------------------------------------------------------

DOMAIN_EXPERT_PROMPT = """You are a Domain Expert (Groundwater Hydrologist) authoring metadata for a Texas Groundwater Availability Model (GAM) dataset to be registered in the SUBSIDE CKAN data catalog.

Your task: produce a complete, schema-conforming metadata object using ONLY the source material provided in the user payload. The source material includes:
- A TWDB landing-page excerpt
- Consolidated findings from the GAM report PDF (map-reduce extraction)
- The model file inventory (resource_plan)
- DIS-derived bounding box / GeoJSON (if available)
- The subside_dataset schema field list with controlled-vocabulary choices
- GAM-specific defaults that MUST be applied
- An optional `organizational_metadata` block with authoritative externally-provided values

OUTPUT FORMAT: Return STRICT JSON only — no markdown, no comments, no trailing commas.

SUBSIDE schema field keys to populate:
{
  "title": string,
  "name": string (lowercase, hyphenated slug),
  "notes": string (2-4 sentences: what the model is, geographic/temporal scope, source org),
  "url": string (TWDB landing page URL),
  "version": string or null,
  "author": string or null,
  "author_email": string or null,
  "maintainer": string or null,
  "maintainer_email": string or null,
  "license_id": string or null,
  "owner_org": string or null,
  "tag_string": string (comma-separated tags for discovery),
  "temporal_coverage_start": string or null (ISO-8601 date or year, e.g. "1980" or "1980-01-01"),
  "temporal_coverage_end": string or null,
  "program_area": string or null,
  "data_contact_email": string or null,
  "caveats_usage": string or null,
  "categories": list[string] (use controlled vocab; MUST include "Groundwater" for GAMs),
  "primary_tags": string or null,
  "secondary_tags": string or null,
  "collection_method": string (MUST be "Model Output" for all GAMs),
  "collection_method_description": string or null,
  "quality_control_level": string or null,
  "quality_assurance_description": string or null,
  "coordinate_system": string or null,
  "spatial": string or null (GeoJSON string if bbox available),
  "update_frequency": string or null,
  "from_date": string or null (ISO-8601),
  "to_date": string or null (ISO-8601),
  "disclaimer": string or null,
  "additional_information": string or null,
  "supporting_url": string or null
}

ORGANIZATIONAL METADATA:
An `organizational_metadata` block (any of: license_id, author, author_email, maintainer,
maintainer_email, owner_org, data_contact_email) may be provided in the inputs. These are
AUTHORITATIVE externally-provided values — populate the corresponding output fields DIRECTLY
from them and DROP any `_gap_` annotation for those fields. Do not invent alternatives. Only
emit a `_gap_<field>` when the field is absent from BOTH the document sources/file_inventory
AND organizational_metadata.

FILE INVENTORY GUIDANCE:
A `file_inventory` (filenames + extension counts) may be provided in the inputs. Treat filenames
and scenario tokens as VALID SOURCE EVIDENCE.

- Use it to infer `temporal_coverage_start`/`temporal_coverage_end` ONLY when filenames/tokens
  clearly indicate a period — e.g. a 'YYYY-YYYY' range, a standalone four-digit year, or scenario
  tokens like 'ss'/'steady-state' (no specific dates), 'tr'/'transient', 'calibration',
  'predictive', 'historical'. If a clear period is present, set the temporal field (ISO-8601 year
  or date) and DROP the corresponding `_gap_` annotation. If there is no clear temporal signal in
  the files or other sources, keep the field null WITH its `_gap_` annotation (do NOT fabricate).
- Use the file inventory to enrich `notes` with a brief description of the model file types/formats
  present and their role (e.g. MODFLOW input packages, output/head files, reports).

MANDATORY RULES — violation of these rules produces unusable metadata:

1. MARK UNKNOWNS: If a field cannot be determined from the source material, set it to null and add a
   companion "_gap_<field>" key with a brief reason, e.g.:
   "temporal_coverage_start": null, "_gap_temporal_coverage_start": "no date found in sources"
   Do NOT guess, fabricate, or extrapolate values not in the source material.

2. APPLY GAM DEFAULTS unconditionally:
   - "collection_method": "Model Output"
   - "categories": must include "Groundwater" (may include others from controlled vocab)

3. Do NOT invent authors, emails, spatial extents, or temporal dates not present in the sources.

4. "name" must be lowercase, URL-safe, hyphen-separated (no spaces, no special chars).

5. "tag_string" must be comma-separated (e.g., "groundwater, aquifer, modflow, texas").

Controlled vocabulary for "categories": Boundaries, Groundwater, Natural Hazards, Planning, Water Quality, Water Use
Controlled vocabulary for "collection_method": Administrative Record, Instrumentation Measurement, Imagery, Human Collected Observation, Survey, Model Output, Geocoding, Digitization, Ground Survey, Analysis or Synthesis, GPS Measurement, Unknown, Other
"""

DATA_CURATOR_PROMPT = """You are a Data Curator evaluating CKAN dataset metadata against FAIR data principles
(Findable, Accessible, Interoperable, Reusable).

Your task: review the candidate metadata object in the user payload and return a structured verdict.

FAIR criteria to check:
- Findable: persistent identifiers referenced, rich/searchable metadata present, tags/categories populated, title is descriptive and discovery-friendly.
- Accessible: resource URLs resolvable or marked as link-type, data format/protocol declared, data contact email provided or explicitly unavailable.
- Interoperable: controlled-vocab terms used (not free-text alternatives), temporal fields in ISO-8601, CRS declared alongside spatial, MINT variables present per resource if applicable.
- Reusable: license present or explicitly unavailable, provenance/lineage traceable from notes, caveats/usage populated or explicitly unavailable, notes/abstract sufficient for reuse without further context.

OUTPUT FORMAT: Return STRICT JSON only:
{
  "verdict": "pass" | "revise",
  "questions": [list of specific questions the author must resolve — one per blocking issue],
  "recommendations": [list of non-blocking improvement suggestions]
}

verdict = "pass" means no BLOCKING FAIR issues remain (non-blocking suggestions are listed but do not block pass).
verdict = "revise" means at least one blocking FAIR issue exists.

CRITICAL CONSTRAINT — Re-raise prevention:
When a field is null and carries a `_gap_<field>` annotation (e.g., "_gap_temporal_coverage_start": "no date found in sources"),
it is genuinely unavailable from the dataset's sources. ACKNOWLEDGE it as unavailable and DO NOT raise a question or a
recommendation asking to provide, add, or 'consider adding' it — doing so is noise. Reserve questions/recommendations
for fields that are PRESENT but improvable, or that are derivable from the provided sources/file_inventory but were missed.
Never ask the author to supply information that cannot come from the dataset's sources or organizational metadata
(e.g., a license the publisher never stated, a maintainer that does not exist).
Only raise questions about fields that are MISSING a value AND do not have a _gap annotation, or where the provided
value is clearly incorrect.

DATA FORMAT NOTE:
Per-resource data format is assigned by CKAN automatically from file extensions at registration time. Do NOT raise
'declare a data format' as a dataset-level gap or blocking issue. If useful, the author may summarize file types
in notes from the file_inventory, but this is a non-blocking suggestion only.
"""

DATA_SCIENTIST_PROMPT = """You are a Data Scientist evaluating whether a domain-knowledgeable researcher could understand
and use this dataset without any further context beyond the metadata.

Your task: review the candidate metadata object in the user payload and return a structured verdict.

Usability criteria to check:
- Abstract/notes answers: what is this model, what aquifer/region/system does it represent, what is the geographic and temporal scope.
- Variables and units are explained or can be inferred; acronyms are expanded on first use (e.g., "GAM" should be spelled out as "Groundwater Availability Model").
- Temporal extent (from_date / to_date or temporal_coverage_start/end) is clearly stated or explicitly marked unavailable.
- Spatial extent is clearly stated and tied to a named aquifer or geographic region (Texas county, basin, etc.).
- File roles and formats are understandable from resource names and descriptions.
- No unexplained jargon that would block a competent hydrologist unfamiliar with TWDB conventions.
- The dataset is distinguishable from other GAMs — generic titles like "Groundwater Model" without a named aquifer fail this check.

OUTPUT FORMAT: Return STRICT JSON only:
{
  "verdict": "pass" | "revise",
  "questions": [list of specific questions the author must resolve — one per blocking issue],
  "recommendations": [list of non-blocking improvement suggestions]
}

verdict = "pass" means the metadata is usable as-is for a data scientist (non-blocking suggestions listed but do not block pass).
verdict = "revise" means at least one usability issue would prevent a competent user from understanding or finding the data.

CRITICAL CONSTRAINT — Re-raise prevention:
When a field is null and carries a `_gap_<field>` annotation (e.g., "_gap_temporal_coverage_start": "no date found in sources"),
it is genuinely unavailable from the dataset's sources. ACKNOWLEDGE it as unavailable and DO NOT raise a question or a
recommendation asking to provide, add, or 'consider adding' it — doing so is noise. Reserve questions/recommendations
for fields that are PRESENT but improvable, or that are derivable from the provided sources/file_inventory but were missed.
Never ask the author to supply information that cannot come from the dataset's sources or organizational metadata
(e.g., a license the publisher never stated, a maintainer that does not exist).
Only raise questions about fields that are MISSING a value AND do not have a _gap annotation, or where the provided
value is clearly incorrect.

DATA FORMAT NOTE:
Per-resource data format is assigned by CKAN automatically from file extensions at registration time. Do NOT raise
'declare a data format' as a dataset-level gap or blocking issue. If useful, the author may summarize file types
in notes from the file_inventory, but this is a non-blocking suggestion only.
"""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorVerdict:
    """Result from a single evaluator persona for a single round.

    verdict: "pass" means no blocking issues; "revise" means at least one.
    questions: blocking issues the author must address.
    recommendations: non-blocking suggestions.
    """
    verdict: Literal["pass", "revise"]
    questions: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# Spec name is EvaluatorResult in D3; EvaluatorVerdict is used in the task description.
# Both names are exported; EvaluatorResult is an alias for backward-compat with the spec's D3.
EvaluatorResult = EvaluatorVerdict


@dataclass
class LoopRound:
    """Transcript entry for a single round of the persona loop."""
    round_number: int
    candidate_metadata: dict[str, Any]
    fair_evaluator: EvaluatorVerdict
    usability_evaluator: EvaluatorVerdict
    converged: bool


@dataclass
class PersonaLoopResult:
    """Result of the full persona metadata loop.

    converged: True if both evaluators passed in the same round.
    rounds: number of rounds completed (0 if author call failed in round 1).
    proposed_metadata: the last candidate metadata dict produced.
    transcript: list of LoopRound entries (one per completed round).
    stop_reason: "converged" | "max_rounds" | "llm_error"
    model_id: identifier for the model/dataset being registered.
    timestamp: ISO-8601 timestamp string (passed in; not generated internally if avoidable).
    """
    converged: bool
    rounds: int
    proposed_metadata: dict[str, Any]
    transcript: list[LoopRound]
    stop_reason: str
    model_id: str = ""
    timestamp: str = ""
    # Alias for spec D3 compatibility.
    @property
    def rounds_to_converge(self) -> int | None:
        return self.rounds if self.converged else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sleep() -> None:
    """Sleep LLM_CALL_DELAY_SECONDS between LLM calls (re-reads env each call)."""
    delay = float(os.environ.get("LLM_CALL_DELAY_SECONDS", str(LLM_CALL_DELAY_SECONDS)))
    if delay > 0:
        time.sleep(delay)


def _extract_resolved_gaps(candidate_metadata: dict[str, Any]) -> dict[str, str]:
    """Extract _gap_<field> annotations from a candidate metadata dict.

    Returns {field_key: gap_reason} for all fields that the author marked as
    not-available-in-sources, so evaluators are not re-raised the same issues.
    """
    gaps: dict[str, str] = {}
    for key, value in candidate_metadata.items():
        if key.startswith("_gap_"):
            field_key = key[len("_gap_"):]
            gaps[field_key] = str(value)
    return gaps


def _format_resolved_gaps_section(gaps: dict[str, str]) -> str:
    """Format resolved gaps dict as a prompt section string."""
    if not gaps:
        return ""
    lines = ["Previously resolved gaps — do not re-raise:"]
    for field_key, reason in gaps.items():
        lines.append(f"  - {field_key}: {reason}")
    return "\n".join(lines)


def _call_author(
    *,
    consolidated_inputs: dict[str, Any],
    resource_plan: list[dict[str, Any]],
    mint_standard_variables: list[str] | None,
    bbox_geojson: str | None,
    subside_schema_fields: list[dict[str, Any]],
    gam_defaults: dict[str, Any],
    prior_evaluator_feedback: list[dict[str, Any]] | None,
    resolved_gaps_section: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    file_inventory: dict[str, Any] | None = None,
    organizational_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the Domain Expert (Author) persona and parse the JSON response."""
    user_payload: dict[str, Any] = {
        "consolidated_inputs": consolidated_inputs,
        "resource_plan_summary": [
            {
                "resource_name": r.get("resource_name", ""),
                "format": r.get("format", ""),
                "relative_path": r.get("relative_path", ""),
                "mint_standard_variables": r.get("mint_standard_variables", ""),
            }
            for r in (resource_plan or [])[:30]
        ],
        "resource_count": len(resource_plan or []),
        "mint_standard_variables": mint_standard_variables or [],
        "bbox_geojson": bbox_geojson,
        "subside_schema_fields": subside_schema_fields or [],
        "gam_defaults": gam_defaults or {},
    }

    if file_inventory:
        user_payload["file_inventory"] = file_inventory

    if organizational_metadata:
        user_payload["organizational_metadata"] = organizational_metadata

    if prior_evaluator_feedback:
        user_payload["evaluator_feedback_from_prior_round"] = prior_evaluator_feedback

    if resolved_gaps_section:
        user_payload["resolved_gaps_instruction"] = resolved_gaps_section

    content = _chat_completion_content(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        system_prompt=DOMAIN_EXPERT_PROMPT,
        user_payload=user_payload,
        temperature=0.1,
    )
    return _parse_llm_json(content)


def _call_evaluator(
    *,
    persona: Literal["fair", "usability"],
    candidate_metadata: dict[str, Any],
    resolved_gaps_section: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
) -> EvaluatorVerdict:
    """Call one evaluator persona (FAIR or Usability) and parse the verdict."""
    prompt = DATA_CURATOR_PROMPT if persona == "fair" else DATA_SCIENTIST_PROMPT

    user_payload: dict[str, Any] = {
        "candidate_metadata": candidate_metadata,
    }
    if resolved_gaps_section:
        user_payload["resolved_gaps_instruction"] = resolved_gaps_section

    content = _chat_completion_content(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        system_prompt=prompt,
        user_payload=user_payload,
        temperature=0.1,
    )
    parsed = _parse_llm_json(content)

    raw_verdict = str(parsed.get("verdict", "")).strip().lower()
    verdict: Literal["pass", "revise"] = "pass" if raw_verdict == "pass" else "revise"

    questions = parsed.get("questions") or []
    recommendations = parsed.get("recommendations") or []
    if not isinstance(questions, list):
        questions = [str(questions)]
    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)]

    return EvaluatorVerdict(
        verdict=verdict,
        questions=[str(q) for q in questions],
        recommendations=[str(r) for r in recommendations],
    )


def _run_evaluators_parallel(
    *,
    candidate_metadata: dict[str, Any],
    resolved_gaps_section: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
) -> tuple[EvaluatorVerdict, EvaluatorVerdict]:
    """Run both evaluators in parallel via ThreadPoolExecutor.

    Returns (fair_verdict, usability_verdict). Order of completion is
    non-deterministic; futures are identified by their submitted persona key.
    """
    results: dict[str, EvaluatorVerdict] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _call_evaluator,
                persona="fair",
                candidate_metadata=candidate_metadata,
                resolved_gaps_section=resolved_gaps_section,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            ): "fair",
            executor.submit(
                _call_evaluator,
                persona="usability",
                candidate_metadata=candidate_metadata,
                resolved_gaps_section=resolved_gaps_section,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            ): "usability",
        }
        for future in as_completed(futures):
            persona_key = futures[future]
            results[persona_key] = future.result()  # may raise — caught by caller

    return results["fair"], results["usability"]


def _write_audit_trail(
    result: PersonaLoopResult,
    runs_dir: Path,
) -> None:
    """Serialize the PersonaLoopResult to runs/<model_id>_<timestamp>.json."""
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{result.model_id}_{result.timestamp}.json".replace(" ", "_").replace(":", "")
        output_path = runs_dir / filename

        def _to_serializable(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)

        payload = {
            "model_id": result.model_id,
            "timestamp": result.timestamp,
            "converged": result.converged,
            "rounds": result.rounds,
            "stop_reason": result.stop_reason,
            "proposed_metadata": result.proposed_metadata,
            "transcript": [
                {
                    "round_number": r.round_number,
                    "candidate_metadata": r.candidate_metadata,
                    "fair_evaluator": asdict(r.fair_evaluator),
                    "usability_evaluator": asdict(r.usability_evaluator),
                    "converged": r.converged,
                }
                for r in result.transcript
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2, default=_to_serializable))
        logger.info("Audit trail written to %s", output_path)
    except Exception as exc:
        logger.warning("Failed to write audit trail: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_persona_metadata_loop(
    consolidated_inputs: dict[str, Any],
    resource_plan: list[dict[str, Any]] | None = None,
    mint_standard_variables: list[str] | None = None,
    bbox_geojson: str | None = None,
    subside_schema_fields: list[dict[str, Any]] | None = None,
    gam_defaults: dict[str, Any] | None = None,
    *,
    max_rounds: int = 3,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    model_id: str = "",
    run_timestamp: str = "",
    runs_dir: Path | None = None,
    file_inventory: dict[str, Any] | None = None,
    organizational_metadata: dict[str, Any] | None = None,
) -> PersonaLoopResult:
    """Run the three-persona authoring and evaluation loop.

    Parameters
    ----------
    consolidated_inputs:
        Map-reduce consolidated metadata dict from pdf_extract / Capability B.
        This is the starting material for the Domain Expert author.
    resource_plan:
        List of resource dicts (already annotated with MINT standard variables).
    mint_standard_variables:
        Dataset-level MINT variable names (resource-level annotations are in resource_plan).
    bbox_geojson:
        DIS-derived (or aquifer-fallback) GeoJSON bbox string, or None.
    subside_schema_fields:
        List of field dicts from the subside_dataset YAML schema (for author context).
    gam_defaults:
        Dict of hard GAM defaults to apply (e.g. {"collection_method": "Model Output",
        "categories": ["Groundwater"]}).
    max_rounds:
        Hard cap on evaluation rounds (default 3).
    llm_model:
        LLM model identifier (e.g. "Meta-Llama-3.3-70B-Instruct").
    llm_api_key:
        API key for the LLM endpoint.
    llm_base_url:
        Base URL for the LLM endpoint (e.g. "https://ai.tejas.tacc.utexas.edu").
    model_id:
        Identifier for this dataset (used in audit trail filename).
    run_timestamp:
        ISO-8601 timestamp string for the audit trail filename; if empty,
        a timestamp is generated internally.
    runs_dir:
        Directory for audit trail output. Defaults to Path("runs/") relative to cwd.
    file_inventory:
        Optional compact inventory of model files built by the orchestrator.
        Dict with keys ``file_count`` (int), ``extension_counts`` (dict),
        ``filenames`` (list[str], capped to 200), and optionally
        ``filenames_truncated`` (bool).  When non-empty it is injected into the
        Domain Expert author's ``user_payload`` under ``"file_inventory"`` so the
        LLM can infer temporal coverage from filename/scenario tokens.
    organizational_metadata:
        Optional dict of authoritative externally-provided organizational values.
        Expected keys (all optional): ``license_id``, ``author``, ``author_email``,
        ``maintainer``, ``maintainer_email``, ``owner_org``, ``data_contact_email``.
        When non-empty it is injected into the Domain Expert author's ``user_payload``
        under ``"organizational_metadata"`` on every round so the author treats these
        values as authoritative and drops any ``_gap_`` annotation for those fields.

    Returns
    -------
    PersonaLoopResult
        Always returns (never raises). On LLM failure: converged=False,
        stop_reason="llm_error", proposed_metadata=best-available prior state.
    """
    if not run_timestamp:
        import datetime
        run_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    _runs_dir = runs_dir if runs_dir is not None else Path("runs")
    _resource_plan = resource_plan or []
    _gam_defaults = gam_defaults or {"collection_method": "Model Output", "categories": ["Groundwater"]}
    _subside_schema_fields = subside_schema_fields or []

    transcript: list[LoopRound] = []
    prior_evaluator_feedback: list[dict[str, Any]] | None = None
    last_candidate: dict[str, Any] = {}
    # resolved_gaps_section tracks gaps from the PRIOR round's candidate metadata.
    # It is injected into:
    #   (a) the CURRENT round's author prompt (so the author knows what is already acknowledged)
    #   (b) the CURRENT round's evaluator prompts (so evaluators don't re-raise resolved gaps)
    # It is set from the PRIOR round candidate's _gap_ annotations at end-of-round.
    resolved_gaps_section: str = ""

    for round_num in range(1, max_rounds + 1):
        logger.info("[persona_loop] Round %d/%d — model_id=%r", round_num, max_rounds, model_id)

        # ------------------------------------------------------------------
        # Step 1: Domain Expert authors / revises candidate metadata.
        # resolved_gaps_section here contains prior-round gaps (empty for round 1).
        # ------------------------------------------------------------------
        try:
            candidate_metadata = _call_author(
                consolidated_inputs=consolidated_inputs,
                resource_plan=_resource_plan,
                mint_standard_variables=mint_standard_variables,
                bbox_geojson=bbox_geojson,
                subside_schema_fields=_subside_schema_fields,
                gam_defaults=_gam_defaults,
                prior_evaluator_feedback=prior_evaluator_feedback,
                resolved_gaps_section=resolved_gaps_section,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                file_inventory=file_inventory,
                organizational_metadata=organizational_metadata or None,
            )
        except Exception as exc:
            logger.error(
                "[persona_loop] LLM failure — round=%d persona=domain_expert model_id=%r: %s",
                round_num,
                model_id,
                exc,
            )
            best_available = last_candidate if last_candidate else consolidated_inputs
            result = PersonaLoopResult(
                converged=False,
                rounds=round_num - 1,
                proposed_metadata=best_available,
                transcript=transcript,
                stop_reason="llm_error",
                model_id=model_id,
                timestamp=run_timestamp,
            )
            _write_audit_trail(result, _runs_dir)
            return result

        _sleep()
        last_candidate = candidate_metadata

        # For evaluators in this round, compute the gaps section from the CURRENT
        # candidate's _gap_ annotations. Additionally include any prior-round gaps
        # (accumulated across all previous rounds) so evaluators never re-raise a
        # field that the author acknowledged as unavailable in ANY prior round.
        current_gaps = _extract_resolved_gaps(candidate_metadata)
        # Build an accumulated gaps section by merging prior-round gap text with
        # the current candidate's gaps.  Prior-round gaps are already formatted in
        # resolved_gaps_section; we merge by rebuilding from scratch when both exist.
        if current_gaps and resolved_gaps_section:
            # Parse prior gaps back out (they are stored as formatted text).
            # Re-format the merged set from current candidate gaps; the prior gaps
            # text is appended as a note so evaluators see the full history.
            evaluator_resolved_gaps_section = (
                _format_resolved_gaps_section(current_gaps)
                + "\n"
                + resolved_gaps_section
            )
        elif current_gaps:
            evaluator_resolved_gaps_section = _format_resolved_gaps_section(current_gaps)
        else:
            # No new gaps in current candidate; carry forward prior-round section.
            evaluator_resolved_gaps_section = resolved_gaps_section

        # ------------------------------------------------------------------
        # Step 2: Run both evaluators IN PARALLEL via ThreadPoolExecutor.
        # evaluator_resolved_gaps_section contains gaps from the current AND prior
        # candidate(s), giving evaluators the full history of acknowledged gaps.
        # ------------------------------------------------------------------
        try:
            fair_verdict, usability_verdict = _run_evaluators_parallel(
                candidate_metadata=candidate_metadata,
                resolved_gaps_section=evaluator_resolved_gaps_section,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )
        except Exception as exc:
            logger.error(
                "[persona_loop] LLM failure — round=%d persona=evaluators model_id=%r: %s",
                round_num,
                model_id,
                exc,
            )
            result = PersonaLoopResult(
                converged=False,
                rounds=round_num,
                proposed_metadata=last_candidate,
                transcript=transcript,
                stop_reason="llm_error",
                model_id=model_id,
                timestamp=run_timestamp,
            )
            _write_audit_trail(result, _runs_dir)
            return result

        _sleep()

        # ------------------------------------------------------------------
        # Step 3: Check convergence.
        # ------------------------------------------------------------------
        converged = fair_verdict.verdict == "pass" and usability_verdict.verdict == "pass"

        loop_round = LoopRound(
            round_number=round_num,
            candidate_metadata=candidate_metadata,
            fair_evaluator=fair_verdict,
            usability_evaluator=usability_verdict,
            converged=converged,
        )
        transcript.append(loop_round)

        if converged:
            logger.info(
                "[persona_loop] Converged after %d round(s) — model_id=%r",
                round_num,
                model_id,
            )
            result = PersonaLoopResult(
                converged=True,
                rounds=round_num,
                proposed_metadata=candidate_metadata,
                transcript=transcript,
                stop_reason="converged",
                model_id=model_id,
                timestamp=run_timestamp,
            )
            _write_audit_trail(result, _runs_dir)
            return result

        # ------------------------------------------------------------------
        # Step 4: Prepare revision context for next round.
        # ------------------------------------------------------------------
        prior_evaluator_feedback = []
        for persona_name, verdict in [("fair_evaluator", fair_verdict), ("usability_evaluator", usability_verdict)]:
            if verdict.verdict == "revise":
                prior_evaluator_feedback.append(
                    {
                        "persona": persona_name,
                        "verdict": verdict.verdict,
                        "questions": verdict.questions,
                        "recommendations": verdict.recommendations,
                    }
                )

        # Extract gaps from current candidate for injection into round 2+ prompts.
        resolved_gaps = _extract_resolved_gaps(candidate_metadata)
        resolved_gaps_section = _format_resolved_gaps_section(resolved_gaps)

        if round_num == max_rounds:
            # Hard cap reached — did not converge.
            logger.warning(
                "[persona_loop] Hard cap of %d rounds reached without convergence — model_id=%r. "
                "Outstanding questions: FAIR=%r, Usability=%r",
                max_rounds,
                model_id,
                fair_verdict.questions,
                usability_verdict.questions,
            )
            result = PersonaLoopResult(
                converged=False,
                rounds=round_num,
                proposed_metadata=candidate_metadata,
                transcript=transcript,
                stop_reason="max_rounds",
                model_id=model_id,
                timestamp=run_timestamp,
            )
            _write_audit_trail(result, _runs_dir)
            return result

    # Should not be reached given the hard-cap check above, but just in case.
    result = PersonaLoopResult(
        converged=False,
        rounds=max_rounds,
        proposed_metadata=last_candidate,
        transcript=transcript,
        stop_reason="max_rounds",
        model_id=model_id,
        timestamp=run_timestamp,
    )
    _write_audit_trail(result, _runs_dir)
    return result
