"""Batched LLM-driven resource-description reviewer for GAM/MODFLOW CKAN datasets.

Combines multiple resource entries into one JSON payload per LLM call (a "Data
Curator" reviewer) to improve their descriptions while keeping per-file LLM
calls low and avoiding 429 rate-limit errors.

Public API
----------
review_resource_descriptions(resource_plan, *, llm_model, llm_api_key,
                              llm_base_url=None, batch_size=25,
                              max_batches=None, dataset_context="") -> int

The function mutates ``resource_plan`` in place, updating only
``resource_description`` for each entry where the LLM returned a non-empty
improved description.  ``resource_name`` and ``resource_title`` are NEVER
changed.

Returns the number of descriptions that were updated.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the Data Curator persona
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Data Curator reviewing MODFLOW/GAM (Groundwater Availability Model) \
model-file resources for a CKAN dataset.

Your task: for each resource in the input list, improve its description to be \
accurate, concise (1-2 sentences), and useful for data discovery.

Ground your improvements in the file name, file extension meaning, and the \
provided dataset_context. MUST NOT invent facts. If you are unsure about a \
file's specific role, keep or lightly refine the current description.

Return STRICT JSON only — no markdown, no comments, no trailing commas.
Required format:
{
  "resources": [
    {"file_name": "<exactly as given>", "description": "<improved description>"},
    ...
  ]
}

Every input file_name must appear in your response, preserving the file_name \
value exactly as it was given.\
"""

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def review_resource_descriptions(
    resource_plan: list[dict[str, Any]],
    *,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None = None,
    batch_size: int = 25,
    max_batches: int | None = None,
    dataset_context: str = "",
) -> int:
    """Improve resource descriptions in *resource_plan* using one LLM call per batch.

    Parameters
    ----------
    resource_plan:
        List of resource dicts, each containing at minimum ``resource_name``,
        ``resource_title``, ``resource_description``, and ``relative_path``.
        Mutated in place — only ``resource_description`` is updated.
    llm_model:
        Model identifier string (e.g. ``"Meta-Llama-3.3-70B-Instruct"``).
    llm_api_key:
        API key / Bearer token for the LLM endpoint.  If empty/falsy, returns 0
        immediately (no-op; deterministic descriptions stand).
    llm_base_url:
        Optional custom base URL for HTTP-fallback LLM calls.
    batch_size:
        Number of resources per LLM call (default 25).
    max_batches:
        If set, process only the first *max_batches* batches.  Resources in
        remaining batches keep their deterministic descriptions and a WARNING is
        logged (no silent truncation).
    dataset_context:
        Free-text context about the dataset (e.g. model title / package_id or
        landing-page excerpt) to help the LLM write better descriptions.

    Returns
    -------
    int
        Number of descriptions actually updated (improved by the LLM).
    """
    # Lazy/defensive imports — keep module importable without optional deps.
    from . import utils as _u  # noqa: F401

    # Guard: empty api_key or empty resource_plan → no-op.
    cleaned_key = _u.clean_text(llm_api_key)
    if not cleaned_key or not resource_plan:
        return 0

    batch_size = max(1, int(batch_size))

    # Build list of batches.
    batches: list[list[dict[str, Any]]] = []
    for start in range(0, len(resource_plan), batch_size):
        batches.append(resource_plan[start : start + batch_size])

    # Apply max_batches cap.
    skipped_resources = 0
    if max_batches is not None and max_batches < len(batches):
        skipped_resources = sum(len(b) for b in batches[max_batches:])
        batches = batches[:max_batches]
        logger.warning(
            "resource_review: max_batches=%d reached; %d resource(s) will keep "
            "their deterministic descriptions (not reviewed by LLM).",
            max_batches,
            skipped_resources,
        )

    inter_call_delay = float(os.environ.get("LLM_CALL_DELAY_SECONDS", "4"))

    total_updated = 0

    for batch_idx, batch in enumerate(batches):
        # Sleep between batches (not after the last one).
        if batch_idx > 0:
            time.sleep(inter_call_delay)

        updated = _review_batch(
            batch,
            batch_idx=batch_idx,
            llm_model=llm_model,
            llm_api_key=cleaned_key,
            llm_base_url=llm_base_url,
            dataset_context=dataset_context,
        )
        total_updated += updated

    return total_updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_batch_payload(
    batch: list[dict[str, Any]],
    dataset_context: str,
) -> dict[str, Any]:
    """Build the user_payload dict for one batch."""
    resources = []
    for item in batch:
        file_name = (
            _u.clean_text(item.get("relative_path"))
            or _u.clean_text(item.get("resource_name"))
        )
        resources.append(
            {
                "file_name": file_name,
                "title": _u.clean_text(item.get("resource_title")),
                "current_description": _u.clean_text(item.get("resource_description")),
            }
        )
    return {
        "dataset_context": dataset_context,
        "resources": resources,
    }


def _review_batch(
    batch: list[dict[str, Any]],
    *,
    batch_idx: int,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str | None,
    dataset_context: str,
) -> int:
    """Run one LLM call for *batch* and update descriptions in place.

    Returns the number of descriptions updated.  On any failure, logs a
    warning and returns 0 (deterministic descriptions are preserved).
    """
    # Lazy import — safe to call after module loads.
    from . import utils as _u  # noqa: F401 (re-import for clarity in isolated fn)

    payload = _build_batch_payload(batch, dataset_context)

    # Build a lookup: file_name -> resource_plan item.
    # We use the same key we put in the payload for round-tripping.
    fn_to_item: dict[str, dict[str, Any]] = {}
    for item, entry in zip(batch, payload["resources"]):
        fn_to_item[entry["file_name"]] = item

    try:
        raw = _u._chat_completion_content(
            model=llm_model,
            api_key=llm_api_key,
            system_prompt=_SYSTEM_PROMPT,
            user_payload=payload,
            base_url=llm_base_url,
            temperature=0.1,
        )
    except Exception as exc:
        logger.warning(
            "resource_review: batch %d LLM call failed (%s: %s); "
            "keeping deterministic descriptions for %d resource(s).",
            batch_idx + 1,
            type(exc).__name__,
            exc,
            len(batch),
        )
        return 0

    # Parse response.
    try:
        parsed = _u._parse_llm_json(raw)
    except Exception as exc:
        logger.warning(
            "resource_review: batch %d JSON parse error (%s); "
            "keeping deterministic descriptions.",
            batch_idx + 1,
            exc,
        )
        return 0

    returned_resources = parsed.get("resources")
    if not isinstance(returned_resources, list):
        logger.warning(
            "resource_review: batch %d response missing 'resources' list; "
            "keeping deterministic descriptions.",
            batch_idx + 1,
        )
        return 0

    updated = 0
    for entry in returned_resources:
        if not isinstance(entry, dict):
            continue
        file_name = _u.clean_text(entry.get("file_name"))
        description = _u.clean_text(entry.get("description"), max_chars=2000)
        if not file_name or not description:
            continue
        item = fn_to_item.get(file_name)
        if item is None:
            logger.warning(
                "resource_review: batch %d returned unknown file_name %r; skipping.",
                batch_idx + 1,
                file_name,
            )
            continue
        item["resource_description"] = description
        updated += 1

    if updated < len(batch):
        missing = len(batch) - updated
        logger.warning(
            "resource_review: batch %d — %d of %d resource(s) not updated by LLM "
            "(missing or unparseable entries); keeping deterministic descriptions.",
            batch_idx + 1,
            missing,
            len(batch),
        )

    return updated


# ---------------------------------------------------------------------------
# Module-level _u reference (populated lazily on first real call, but we need
# the reference available for _build_batch_payload and _review_batch which
# import it themselves).  This just makes the module importable without deps.
# ---------------------------------------------------------------------------
try:
    from . import utils as _u  # noqa: F401
except ImportError:  # pragma: no cover
    _u = None  # type: ignore[assignment]
