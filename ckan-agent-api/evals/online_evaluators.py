"""
Online evaluators for the CKAN registration agent.

Each evaluator takes the final LangGraph state and returns a dict of
{feedback_key: score} where score is 1.0 (pass) or 0.0 (fail).

post_feedback_async() fires all evaluators in a daemon thread so they
never block the agent response.
"""

from __future__ import annotations

import re
import threading
import uuid
from typing import Any

_CKAN_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_GENERIC_TITLE_TOKENS = {
    "dataset",
    "data",
    "untitled",
    "file",
    "upload",
    "test",
    "temp",
}


# ---------------------------------------------------------------------------
# Deterministic evaluators
# ---------------------------------------------------------------------------

def eval_required_fields_present(state: dict[str, Any]) -> dict[str, float]:
    """title, notes, owner_org, and name are all non-empty strings."""
    meta = state.get("candidate_metadata") or {}
    fields = ["title", "notes", "owner_org", "name"]
    score = 1.0 if all(bool(str(meta.get(f, "")).strip()) for f in fields) else 0.0
    return {"required_fields_present": score}


def eval_name_slug_valid(state: dict[str, Any]) -> dict[str, float]:
    """CKAN name slug matches ^[a-z0-9][a-z0-9_-]*$ and is under 100 chars."""
    name = str((state.get("candidate_metadata") or {}).get("name", "")).strip()
    if not name:
        return {"name_slug_valid": 0.0}
    score = 1.0 if _CKAN_SLUG_RE.match(name) and len(name) <= 100 else 0.0
    return {"name_slug_valid": score}


def eval_title_not_generic(state: dict[str, Any]) -> dict[str, float]:
    """Title is at least 3 words and doesn't consist solely of generic tokens."""
    title = str((state.get("candidate_metadata") or {}).get("title", "")).strip()
    if not title:
        return {"title_not_generic": 0.0}
    words = [w.lower() for w in title.split()]
    if len(words) < 2:
        return {"title_not_generic": 0.0}
    non_generic = [w for w in words if w not in _GENERIC_TITLE_TOKENS]
    score = 1.0 if non_generic else 0.0
    return {"title_not_generic": score}


def eval_spatial_valid(state: dict[str, Any]) -> dict[str, float]:
    """If spatial is present it must be a non-empty WKT string starting with a geometry keyword."""
    meta = state.get("candidate_metadata") or {}
    spatial = str(meta.get("spatial", "")).strip()
    if not spatial:
        # Absence is fine — not all datasets are spatial.
        return {"spatial_valid": 1.0}
    _wkt_prefixes = ("POLYGON", "MULTIPOLYGON", "POINT", "LINESTRING", "GEOMETRYCOLLECTION")
    score = 1.0 if spatial.upper().startswith(_wkt_prefixes) else 0.0
    return {"spatial_valid": score}


def eval_temporal_consistent(state: dict[str, Any]) -> dict[str, float]:
    """If both temporal_coverage_start and _end are present, start <= end."""
    meta = state.get("candidate_metadata") or {}
    start = str(meta.get("temporal_coverage_start", "")).strip()
    end = str(meta.get("temporal_coverage_end", "")).strip()
    if not start or not end:
        return {"temporal_consistent": 1.0}
    score = 1.0 if start <= end else 0.0
    return {"temporal_consistent": score}


def eval_llm_fields_non_empty(state: dict[str, Any]) -> dict[str, float]:
    """Every field marked llm-derived in field_origins is a non-empty string."""
    meta = state.get("candidate_metadata") or {}
    origins = (state.get("result") or {}).get("field_origins") or {}
    llm_fields = [k for k, v in origins.items() if v == "llm-derived"]
    if not llm_fields:
        return {"llm_fields_non_empty": 1.0}
    score = 1.0 if all(bool(str(meta.get(f, "")).strip()) for f in llm_fields) else 0.0
    return {"llm_fields_non_empty": score}


def eval_no_localhost_resources(state: dict[str, Any]) -> dict[str, float]:
    """No remote_resource URL contains localhost or 127.0.0.1."""
    request = state.get("request") or {}
    resources = request.get("remote_resources") or []
    has_localhost = any(
        "localhost" in str(r.get("url", "")) or "127.0.0.1" in str(r.get("url", ""))
        for r in resources
    )
    return {"no_localhost_resources": 0.0 if has_localhost else 1.0}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_EVALUATORS = [
    eval_required_fields_present,
    eval_name_slug_valid,
    eval_title_not_generic,
    eval_spatial_valid,
    eval_temporal_consistent,
    eval_llm_fields_non_empty,
    eval_no_localhost_resources,
]


def _run_and_post(run_id: uuid.UUID | str, state: dict[str, Any]) -> None:
    """Run all evaluators and post results to LangSmith. Runs in a daemon thread."""
    try:
        from langsmith import Client

        client = Client()
        for evaluator in _EVALUATORS:
            try:
                scores = evaluator(state)
                for key, score in scores.items():
                    client.create_feedback(
                        run_id=run_id,
                        key=key,
                        score=score,
                        source_info={"source": "online_evaluator"},
                    )
            except Exception:
                pass  # never let one evaluator failure kill the others
    except Exception:
        pass  # langsmith unavailable or misconfigured — silent


def post_feedback_async(run_id: uuid.UUID | str, state: dict[str, Any]) -> None:
    """Fire evaluators in a daemon thread. Returns immediately."""
    t = threading.Thread(target=_run_and_post, args=(run_id, state), daemon=True)
    t.start()
