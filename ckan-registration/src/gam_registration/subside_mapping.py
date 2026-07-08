"""SUBSIDE dataset field mapping for GAM CKAN registrations.

Built to the LIVE 15-column ``subside_dataset`` schema deployed on
ckan.tacc.utexas.edu (confirmed via scheming_dataset_schema_show, 2026-06-25).

The deployed schema defines exactly these dataset-level COLUMNS:
    title, name, notes, tag_string (→ tags list), license_id, owner_org,
    url, version, author, author_email, maintainer, maintainer_email,
    temporal_coverage_start, temporal_coverage_end, mint_standard_variables.

IMPORTANT PLACEMENT NOTE — this reverses the earlier proposed-extension design:
  - The earlier spec (OQ-1 / OQ-2 resolved 2026-06-25) placed classification
    and spatial fields as dataset-level columns and mint_standard_variables as
    a resource-level field, anticipating a schema redeploy.
  - The LIVE deployed schema does NOT include those classification columns at
    the dataset level and does NOT carry them at the resource level either.
  - We will NOT redeploy the schema.  Instead, extra SUBSIDE fields
    (categories, collection_method, spatial, program_area, caveats_usage,
    primary_tags, secondary_tags, quality_control_level, data_contact_email,
    and other free-form metadata) are stored as CKAN dataset ``extras``
    (list of {"key": ..., "value": ...}).
  - mint_standard_variables is a DATASET COLUMN (multiple_text field in the
    live schema), NOT a resource-level field.

GAM-specific defaults applied unconditionally (land in extras):
  - collection_method = "Model Output"
  - categories = ["Groundwater"]    (list → JSON array string in extras)
"""

from __future__ import annotations

import calendar
import json
import logging
import re
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_iso_date(value: Any, *, is_end: bool) -> str | None:
    """Normalize a temporal value to ISO ``YYYY-MM-DD`` for CKAN's date preset.

    CKAN's scheming ``date`` preset rejects bare years / ranges ("Date format
    incorrect").  Accepts full ISO dates (validated, passed through), ``YYYY-MM``
    (expanded to first/last day of month), or any string containing 4-digit
    year token(s) (expanded to Jan 1 for a start field, Dec 31 for an end
    field).  Returns ``None`` when no usable date can be derived, so the field
    is OMITTED rather than failing validation.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Full ISO date (optionally with a time component) -> validate, keep date part.
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass
    # YYYY-MM -> first / last day of the month.
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            d = calendar.monthrange(y, mo)[1] if is_end else 1
            return date(y, mo, d).isoformat()
    # Any 4-digit year token(s) -> Jan 1 (start) / Dec 31 (end).
    years = [int(y) for y in re.findall(r"\d{4}", s) if 1000 <= int(y) <= 2999]
    if years:
        y = max(years) if is_end else min(years)
        return f"{y:04d}-12-31" if is_end else f"{y:04d}-01-01"
    return None

# Default values applied to every GAM registration.
_GAM_DEFAULTS: dict[str, Any] = {
    "collection_method": "Model Output",
    "categories": ["Groundwater"],
}

# The 15 live dataset-level COLUMNS in the deployed subside_dataset schema.
_LIVE_DATASET_COLUMNS = {
    "type", "name", "title", "notes", "tags", "license_id", "owner_org",
    "url", "version", "author", "author_email", "maintainer", "maintainer_email",
    "temporal_coverage_start", "temporal_coverage_end", "mint_standard_variables",
}

# Extra SUBSIDE fields stored as CKAN dataset extras (not schema columns).
# Order is preserved in the extras list.
_EXTRAS_FIELDS = [
    "categories",
    "collection_method",
    "program_area",
    "caveats_usage",
    "primary_tags",
    "secondary_tags",
    "quality_control_level",
    "data_contact_email",
    "spatial",
    "publishing_status",
    "coordinate_system",
    "coordinate_field_format",
    "collection_method_description",
    "quality_assurance_description",
    "update_type",
    "update_frequency",
    "from_date",
    "to_date",
    "disclaimer",
    "additional_information",
    "related_resources",
    "supporting_url",
]


def _coerce_list_field(value: Any) -> str | None:
    """Coerce a field that may be a list into a JSON string for CKAN extras storage.

    Returns None if the value is empty.
    """
    if value is None:
        return None
    if isinstance(value, list):
        clean = [str(v) for v in value if v is not None and str(v).strip()]
        if not clean:
            return None
        return json.dumps(clean)
    text = str(value).strip()
    return text if text else None


def _clean(value: Any, max_chars: int | None = None) -> str | None:
    """Return a stripped string or None for falsy/whitespace-only values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _encode_extra_value(value: Any) -> str | None:
    """Encode a value for storage in a CKAN extras {key, value} entry.

    - strings: returned as-is (after stripping)
    - lists: JSON array string
    - dicts (e.g. GeoJSON): JSON string
    - None / whitespace-only: returns None (caller should omit the entry)
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, list):
        clean = [str(v) for v in value if v is not None and str(v).strip()]
        if not clean:
            return None
        return json.dumps(clean)
    text = str(value).strip()
    return text if text else None


def map_to_subside_dataset(
    proposed: dict[str, Any],
    *,
    mint_vars: list[str] | None = None,
    spatial: str | None = None,
    owner_org: str | None = None,
    gam_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map a proposed metadata dict to a CKAN package body for subside_dataset.

    Built to the LIVE 15-column subside_dataset schema on ckan.tacc.utexas.edu.
    Extra SUBSIDE fields are stored as CKAN dataset ``extras`` (list of
    {"key": ..., "value": ...}) — no schema redeploy is required.

    This is the third explicit step in the notebook orchestration:
      1. propose_ckan_dataset_metadata_with_llm / map-reduce → proposed dict
      2. run_persona_metadata_loop → PersonaLoopResult.proposed_metadata
      3. map_to_subside_dataset(proposed_metadata, ...) → CKAN package body  <-- here

    Dataset-level COLUMNS (15 live fields):
        title, name, notes, tags, license_id, owner_org, url, version,
        author, author_email, maintainer, maintainer_email,
        temporal_coverage_start, temporal_coverage_end,
        mint_standard_variables (list — populated from ``mint_vars``).

    Dataset ``extras`` (stored as list of {"key": ..., "value": ...}):
        categories, collection_method, program_area, caveats_usage,
        primary_tags, secondary_tags, quality_control_level,
        data_contact_email, spatial (JSON-encoded), publishing_status,
        coordinate_system, coordinate_field_format,
        collection_method_description, quality_assurance_description,
        update_type, update_frequency, from_date, to_date, disclaimer,
        additional_information, related_resources, supporting_url.

    Parameters
    ----------
    proposed:
        Dict from the map-reduce consolidation or persona loop.  Keys may use
        either the LLM's "dataset_*" prefix convention (e.g. "dataset_title")
        or bare CKAN field names (e.g. "title").  Both are handled.
    mint_vars:
        List of MINT standard variable name strings.  Placed at DATASET level
        as ``mint_standard_variables`` (live schema column, multiple_text).
        Omitted from the package body if None or empty.
    spatial:
        Explicit spatial GeoJSON string or dict (from bbox derivation).  If
        provided, takes precedence over any ``spatial`` value in *proposed*.
        Stored in extras as a JSON string.
    owner_org:
        CKAN organization slug.  If provided, included as a top-level column.
    gam_defaults:
        Override the module-level GAM defaults.  Merged on top of the defaults;
        caller-supplied values win.  Pass ``{}`` to suppress defaults entirely.

    Returns
    -------
    dict
        CKAN package dict ready to pass to ``create_or_update_ckan_dataset``
        or ``ckan_action_post("package_create", ...)``.

        Top-level keys (live schema columns only):
            type, name, title, notes, url, tags, author, author_email,
            maintainer, maintainer_email, license_id, version, owner_org,
            temporal_coverage_start, temporal_coverage_end,
            mint_standard_variables.
            Null/empty fields are omitted.

        ``extras`` key:
            List of {"key": ..., "value": ...} for the SUBSIDE-specific fields.
            Present only when at least one extra has a non-empty value.
    """
    defaults = dict(_GAM_DEFAULTS)
    if gam_defaults is not None:
        defaults.update(gam_defaults)

    def _get(key: str, alt_key: str | None = None) -> Any:
        """Retrieve a value from *proposed* by bare key or "dataset_" prefix."""
        value = proposed.get(key)
        if value is None and alt_key:
            value = proposed.get(alt_key)
        if value is None:
            # Try stripping "dataset_" prefix.
            bare = key.removeprefix("dataset_")
            if bare != key:
                value = proposed.get(bare)
        return value

    # --- Core CKAN column fields ---
    title = _clean(_get("dataset_title", "title"), max_chars=140)
    notes = _clean(_get("dataset_notes", "notes"), max_chars=3000)
    url = _clean(_get("dataset_url", "url"), max_chars=500)
    author = _clean(_get("dataset_author", "author"), max_chars=200)
    author_email = _clean(_get("dataset_author_email", "author_email"), max_chars=200)
    maintainer = _clean(_get("dataset_maintainer", "maintainer"), max_chars=200)
    maintainer_email = _clean(_get("dataset_maintainer_email", "maintainer_email"), max_chars=200)
    license_id = _clean(_get("dataset_license_id", "license_id"), max_chars=100)
    version = _clean(_get("dataset_version", "version"), max_chars=100)

    # Derive dataset name (slug).
    raw_name = _get("dataset_name", "name")
    name = _clean(raw_name, max_chars=100)

    # Temporal coverage.
    temporal_start = _normalize_iso_date(_get("temporal_coverage_start"), is_end=False)
    temporal_end = _normalize_iso_date(_get("temporal_coverage_end"), is_end=True)

    # Tags — handle both list-of-dicts (CKAN format) and list-of-strings.
    raw_tags = _get("dataset_tags", "tags")
    tags: list[dict[str, str]] = []
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if isinstance(tag, dict) and tag.get("name"):
                tags.append({"name": str(tag["name"])})
            elif isinstance(tag, str) and tag.strip():
                tags.append({"name": tag.strip()})

    # --- mint_standard_variables — DATASET COLUMN (live schema, multiple_text) ---
    mint_list: list[str] | None = None
    if mint_vars:
        clean_vars = [str(v).strip() for v in mint_vars if v is not None and str(v).strip()]
        if clean_vars:
            mint_list = clean_vars

    # --- Build the dataset-level package dict (columns only) ---
    package: dict[str, Any] = {
        "type": "subside_dataset",
    }

    if name:
        package["name"] = name
    if title:
        package["title"] = title
    if notes:
        package["notes"] = notes
    if url:
        package["url"] = url
    if author:
        package["author"] = author
    if author_email:
        package["author_email"] = author_email
    if maintainer:
        package["maintainer"] = maintainer
    if maintainer_email:
        package["maintainer_email"] = maintainer_email
    if license_id:
        package["license_id"] = license_id
    if version:
        package["version"] = version
    if owner_org:
        package["owner_org"] = str(owner_org).strip()
    if tags:
        package["tags"] = tags
    if temporal_start:
        package["temporal_coverage_start"] = temporal_start
    if temporal_end:
        package["temporal_coverage_end"] = temporal_end
    if mint_list:
        package["mint_standard_variables"] = mint_list

    # --- Build extras ---
    # Resolve each extras field from proposed (applying GAM defaults where applicable).

    # collection_method — apply GAM default if not in proposed.
    collection_method_raw = _get("collection_method")
    if collection_method_raw is None:
        collection_method_val = defaults.get("collection_method")
    else:
        collection_method_val = collection_method_raw
    if not collection_method_val or (isinstance(collection_method_val, str) and not collection_method_val.strip()):
        collection_method_val = defaults.get("collection_method")

    # categories — apply GAM default if not in proposed.
    categories_raw = _get("categories")
    if categories_raw is None:
        categories_val = defaults.get("categories")
    else:
        categories_val = categories_raw
    if not categories_val:
        categories_val = defaults.get("categories")

    # spatial — explicit parameter takes precedence over proposed.
    if spatial is not None and (isinstance(spatial, dict) or str(spatial).strip()):
        spatial_val: Any = spatial
    else:
        spatial_raw = _get("spatial", "dataset_spatial")
        spatial_val = spatial_raw if spatial_raw is not None else None

    # Collect all extras candidates in the canonical order.
    extras_candidates: list[tuple[str, Any]] = [
        ("categories", categories_val),
        ("collection_method", collection_method_val),
        ("program_area", _get("program_area")),
        ("caveats_usage", _get("caveats_usage")),
        ("primary_tags", _get("primary_tags")),
        ("secondary_tags", _get("secondary_tags")),
        ("quality_control_level", _get("quality_control_level")),
        ("data_contact_email", _get("data_contact_email")),
        ("spatial", spatial_val),
        ("publishing_status", _get("publishing_status")),
        ("coordinate_system", _get("coordinate_system")),
        ("coordinate_field_format", _get("coordinate_field_format")),
        ("collection_method_description", _get("collection_method_description")),
        ("quality_assurance_description", _get("quality_assurance_description")),
        ("update_type", _get("update_type")),
        ("update_frequency", _get("update_frequency")),
        ("from_date", _get("from_date")),
        ("to_date", _get("to_date")),
        ("disclaimer", _get("disclaimer")),
        ("additional_information", _get("additional_information")),
        ("related_resources", _get("related_resources")),
        ("supporting_url", _get("supporting_url")),
    ]

    extras: list[dict[str, str]] = []
    for key, raw_value in extras_candidates:
        encoded = _encode_extra_value(raw_value)
        if encoded is not None:
            extras.append({"key": key, "value": encoded})

    if extras:
        package["extras"] = extras

    return package
