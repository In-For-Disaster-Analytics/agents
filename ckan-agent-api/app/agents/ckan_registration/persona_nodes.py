"""Persona-chat subgraph nodes (spec Increment 3).

Wires the persona engine into a LangGraph flow with a human clarification interrupt:

    persona ──(needs_clarification & under cap)──▶ clarify ──▶ persona
       │                                                         (loop)
       └────────────(converged / cap / done)────────────▶ propose ─▶ END

- ``persona`` runs the (sync, registry-driven) engine. The engine self-loops author +
  evaluators and returns ``needs_clarification`` eagerly (R1/R2).
- ``clarify`` reuses the same ``interrupt()`` mechanism as the existing approval node.
  On resume it splits answers (R3): org-level keys are thread-sticky (``org_metadata``);
  everything else is dataset-specific (``dataset_clarifications``) and is folded into the
  authoritative ``organizational_metadata`` passed back to the author.
- ``propose`` emits review markdown that labels each field's origin (R6:
  user-supplied / llm-derived / schema-default) and writes a legacy-compatible
  ``analyzed`` state file so the existing dry-run/apply path can consume it.

This subgraph does not touch the legacy CKAN worker, so it compiles and tests without
the cross-tree worker dependency. Routing the main graph's ``analyze``/``revise`` actions
through it (behind ``settings.persona_chat_enabled``) and consolidating the ``nodes`` LLM
helper onto ``app/llm`` is the next sub-step.
"""

from __future__ import annotations

import difflib
import json
import logging
import mimetypes
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app import llm
from app.agents.ckan_registration.ckan_client import CkanClient
from app.agents.ckan_registration.logging_config import log_node_entry, log_node_exit
from app.agents.ckan_registration.state import CkanRegistrationState
from app.files import build_file_inventory, gather_file_evidence
from app.personas import Persona, PersonaRegistry, run_persona_metadata_loop
from app.personas.engine import STOP_LLM_ERROR, STOP_MAX_ROUNDS, STOP_NEEDS_CLARIFICATION
from app.schemas import SchemaRegistry
from app.schemas.registry import SchemaProfile
from app.settings import Settings
from app.tools import (
    GEO_PERSONA_METADATA_TOOLS,
    PERSONA_BLOCKED_TOOLS,
    CompositeToolExecutor,
    GeoSyncExecutor,
    InProcessToolExecutor,
    MCPToolExecutor,
    ToolError,
    ToolRegistry,
)
from app.tools.mcp_client import get_shared_client

logger = logging.getLogger(__name__)

# Org-level fields are reused across datasets in a thread; everything else is
# dataset-specific and re-asked per dataset (spec R3). CRS (`coordinate_system`) is
# included because a deployment's datasets almost always share one projection — once
# answered, don't re-ask it for the rest of the thread.
ORG_LEVEL_FIELDS = {
    "owner_org", "author", "author_email", "maintainer", "maintainer_email", "data_contact_email", "coordinate_system",
}
# Fields that must hold a syntactically valid email; an invalid answer is rejected (re-asked)
# rather than stored — so a bare username like "wmobley" never lands in CKAN metadata.
EMAIL_FIELDS = {"data_contact_email", "maintainer_email", "author_email"}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# One question per clarify round, so allow more rounds before giving up and proposing.
CLARIFICATION_CAP = 6
# Presentation order for clarification questions. Fields earlier in this mapping are shown
# first so the conversation flows: who → contact → technical → descriptive.
_FIELD_PRIORITY: dict[str, int] = {
    "author": 0,
    "author_email": 1,
    "maintainer": 2,
    "maintainer_email": 3,
    "data_contact_email": 4,
    "coordinate_system": 5,
    "temporal_coverage_start": 10,
    "temporal_coverage_end": 11,
    "tag_string": 20,
    "notes": 21,
}

# ── Meta-question / confusion detection ──────────────────────────────────────
# Matches answers that are questions or confusion signals rather than field values.
# Requires an interrogative start AND a "?" to avoid false positives on names like "What Farms LLC".
_META_INTERROGATIVE_RE = re.compile(r"^\s*(what|how|which|who|where|when)\b", re.I)
_META_CONFUSION_RE = re.compile(
    r"\b(i\s+don'?t\s+know|not\s+sure|i'?m\s+not\s+sure|"
    r"what\s+are\s+my\s+options|give\s+me\s+an?\s+example|"
    r"can\s+you\s+(help|explain|clarify|tell\s+me))\b",
    re.I,
)
# Per-field hints shown when the user asks a meta-question instead of answering.
_FIELD_GUIDANCE_HINTS: dict[str, str] = {
    "author": "the full name of the person or organization that created this dataset — e.g. `Dr. Jane Smith`, `TWDB`, `University of Texas`",
    "maintainer": "the name of whoever maintains this dataset after publication — e.g. `John Smith`, `TWDB Data Team`. Can be the same as the author.",
    "author_email": "the author's email address — e.g. `name@example.org`",
    "maintainer_email": "the maintainer's email address — type `same as author` to reuse the author's email",
    "data_contact_email": "the public contact email for this dataset — type `same as author` to reuse the author's email",
}
# Per-field shortcut hints always appended to clarification questions for these fields.
_FIELD_CONTEXT: dict[str, str] = {
    "maintainer": "This is the person or org responsible for maintaining the dataset after publication — often the same as the author.",
    "maintainer_email": "Type `same as author` to reuse the author's email address.",
    "data_contact_email": "Type `same as author` to reuse the author's email address.",
}


def _valid_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(str(value).strip()))


def _is_meta_question(text: str) -> bool:
    """Return True when the answer looks like a question or confusion signal rather than a field value."""
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    if _META_CONFUSION_RE.search(stripped):
        return True
    return bool(_META_INTERROGATIVE_RE.search(stripped)) and "?" in stripped


def _field_guidance_message(current: dict[str, Any], state: CkanRegistrationState, questions: list[dict[str, Any]]) -> str:
    """Build a guidance re-prompt when the user typed a question instead of a field value."""
    field = str(current.get("field") or "")
    question_text = (current.get("question") or "Please provide this information.").strip()
    hint = _FIELD_GUIDANCE_HINTS.get(field, "")
    parts = [
        "It looks like you have a question — here's some guidance:",
        "",
        f"**{question_text}**",
    ]
    if hint:
        parts.append(f"This field expects {hint}.")
    parts.append("")
    parts.append("Please enter a specific value:")
    return "\n".join(parts)


EngineFn = Callable[..., Any]


def _slugify(text: str, fallback: str = "dataset") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return slug[:90].strip("-") or fallback


def _resource_plan_from_heads(file_heads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a resource plan from file_heads (output of gather_file_evidence)."""
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for head in file_heads:
        path_str = str(head.get("path") or "")
        name = str(head.get("name") or Path(path_str).name)
        ext = str(head.get("extension") or Path(name).suffix.lower())
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        base = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-") or "resource"
        slug, counter = base, 2
        while slug in seen:
            slug = f"{base}-{counter}"
            counter += 1
        seen.add(slug)
        plan.append({
            "resource_name": slug,
            "resource_title": Path(name).stem.replace("-", " ").replace("_", " ").title(),
            "resource_description": "",
            "resource_tags": [],
            "source_url": "",
            "local_path": path_str,
            "relative_path": name,
            "format": ext.lstrip(".").upper() or "BIN",
            "mimetype": mime,
            "size_bytes": int(head.get("size_bytes") or 0),
            "sha256": "",
        })
    return plan


_BOUNDS_KEYWORDS = frozenset({"bounds", "boundary", "extent", "bbox", "envelope"})


def _is_wgs84_coord(lon: float, lat: float) -> bool:
    """Return True when (lon, lat) look like valid WGS84 degrees."""
    return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0


def _collect_wgs84_coords(geo: dict[str, Any]) -> list[tuple[float, float]]:
    """Recursively extract all (lon, lat) pairs from a GeoJSON object that are in WGS84 range."""
    pts: list[tuple[float, float]] = []

    def _from_coord(c: Any) -> None:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            try:
                lon, lat = float(c[0]), float(c[1])
                if _is_wgs84_coord(lon, lat):
                    pts.append((lon, lat))
            except (TypeError, ValueError):
                pass

    def _from_geom(geom: dict[str, Any]) -> None:
        gtype = geom.get("type", "")
        coords = geom.get("coordinates") or []
        if gtype == "Point":
            _from_coord(coords)
        elif gtype in ("LineString", "MultiPoint"):
            for c in coords:
                _from_coord(c)
        elif gtype in ("Polygon", "MultiLineString"):
            for ring in coords:
                for c in ring:
                    _from_coord(c)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for c in ring:
                        _from_coord(c)
        elif gtype == "GeometryCollection":
            for g in geom.get("geometries") or []:
                _from_geom(g)

    gtype = geo.get("type", "")
    if gtype == "FeatureCollection":
        for feat in geo.get("features") or []:
            _from_geom(feat.get("geometry") or {})
    elif gtype == "Feature":
        _from_geom(geo.get("geometry") or {})
    else:
        _from_geom(geo)

    return pts


def _wgs84_bbox_and_centroid_from_geojson(
    geojson_str: str,
) -> tuple[str, tuple[float, float]] | None:
    """Parse a GeoJSON string; return (bbox_polygon_str, (lat, lon)) if it contains WGS84
    coordinates, or None when the file is in a projected CRS or has no usable coordinates."""
    try:
        geo = json.loads(geojson_str)
    except Exception:  # noqa: BLE001
        return None
    pts = _collect_wgs84_coords(geo)
    if len(pts) < 2:
        return None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    w, e = min(lons), max(lons)
    s, n = min(lats), max(lats)
    centroid_lat = (s + n) / 2
    centroid_lon = (w + e) / 2
    bbox = {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }
    return json.dumps(bbox), (centroid_lat, centroid_lon)


# GeoJSON files whose names suggest they contain dataset bounds or shot positions —
# preferred sources for the WGS84 bbox / location hint.
_SPATIAL_KEYWORDS = frozenset({
    "bounds", "boundary", "extent", "bbox", "envelope", "shots", "footprint",
})


def _wgs84_spatial_from_heads(
    file_heads: list[dict[str, Any]],
) -> tuple[str, tuple[float, float]] | None:
    """Scan file_heads for any GeoJSON file with WGS84 coordinates.

    Prefers files whose name matches _SPATIAL_KEYWORDS (bounds, shots, …) but
    falls back to any .geojson file. Returns (bbox_polygon_str, (lat, lon)) for the
    first file that yields usable WGS84 coords, or None.
    """
    def _score(head: dict[str, Any]) -> int:
        name = str(head.get("name") or "").lower()
        ext = str(head.get("extension") or "").lower()
        if ext not in (".geojson", ".json"):
            return -1
        return 1 if any(kw in name for kw in _SPATIAL_KEYWORDS) else 0

    candidates = sorted(
        (h for h in file_heads if _score(h) >= 0),
        key=_score,
        reverse=True,
    )
    for head in candidates:
        path_str = str(head.get("path") or "")
        if not path_str:
            continue
        try:
            with open(path_str, encoding="utf-8") as fp:
                raw = fp.read()
        except Exception:  # noqa: BLE001
            continue
        result = _wgs84_bbox_and_centroid_from_geojson(raw)
        if result is not None:
            logger.info(
                "[geo] WGS84 bbox+centroid from %s | centroid=%.4f,%.4f",
                head.get("name"), result[1][0], result[1][1],
            )
            return result
    logger.warning("[geo] no WGS84 GeoJSON found in file_heads — bbox and location_hint unavailable")
    return None


# Nominatim OSM class/type values that indicate a geographic feature (bay, river, …)
# rather than a settlement. When the top result is one of these we skip to county/state
# so that "Hooper Bay" (the body of water) doesn't shadow "Bethel" (the nearest town).
_GEOCODE_FEATURE_CLASSES = frozenset({"natural", "waterway", "landuse", "leisure"})
_GEOCODE_FEATURE_TYPES = frozenset({
    "bay", "water", "peak", "ridge", "stream", "river", "lake", "sea",
    "sound", "strait", "wetland", "beach", "cliff", "valley", "wood", "forest",
    "scrub", "coastline", "glacier", "tundra",
})


def _reverse_geocode(lat: float, lon: float, timeout: float = 5.0) -> str | None:
    """Return a specific place name for (lat, lon) via Nominatim; None on any failure.

    Tries zoom=14 (neighbourhood/suburb) first, then falls back to zoom=10 (city/town)
    so remote areas that only resolve to a state at coarse zoom still get the most
    specific named place available.
    """
    import ssl
    import urllib.request as urlreq

    def _fetch(zoom: int, ctx: Any) -> dict[str, Any]:
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?format=json&lat={lat:.6f}&lon={lon:.6f}&zoom={zoom}"
        )
        req = urlreq.Request(url, headers={"User-Agent": "ckan-registration-agent/1.0"})
        kw: dict[str, Any] = {"timeout": timeout}
        if ctx is not None:
            kw["context"] = ctx
        with urlreq.urlopen(req, **kw) as resp:
            return json.loads(resp.read().decode())

    ssl_ctx: Any = None
    probe_data: dict[str, Any] | None = None
    try:
        probe_data = _fetch(14, None)  # probe — capture result; if SSL fails we set up ssl_ctx below
    except Exception as exc:  # noqa: BLE001
        exc_str = str(exc).lower()
        if "ssl" not in exc_str and "certificate" not in exc_str:
            return None
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

    def _extract(data: dict[str, Any]) -> str | None:
        addr = data.get("address") or {}
        # Most-specific → least-specific named-place keys
        for key in ("neighbourhood", "suburb", "quarter", "city_district",
                    "city", "town", "village", "hamlet", "municipality", "borough"):
            if addr.get(key):
                state = addr.get("state", "")
                cc = addr.get("country_code", "").upper()
                parts = [addr[key]]
                if state:
                    parts.append(state)
                # Only append country code for non-US addresses (state alone implies US).
                if cc and cc != "US":
                    parts.append(cc)
                return ", ".join(parts)
        # Named feature as a fallback before county/state — but only when it's a settlement
        # or recognizable place, not a raw geographic feature (bay, waterway, …).
        # Nominatim returns class/type on the top-level result object (not inside address).
        name = data.get("name") or ""
        is_geo_feature = (
            data.get("class", "") in _GEOCODE_FEATURE_CLASSES
            or data.get("type", "") in _GEOCODE_FEATURE_TYPES
        )
        if name and not is_geo_feature and name not in (addr.get("state", ""), addr.get("country", "")):
            state = addr.get("state", "")
            parts = [name] + ([state] if state else [])
            return ", ".join(parts)
        county = addr.get("county", "")
        state = addr.get("state", "")
        parts = [p for p in (county, state) if p]
        return ", ".join(parts) if parts else None

    # Try zoom=14 first; if result is only county/state level, retry at zoom=10.
    for zoom in (14, 10):
        try:
            data = probe_data if zoom == 14 and probe_data is not None else _fetch(zoom, ssl_ctx)
        except Exception:  # noqa: BLE001
            return None
        result = _extract(data)
        if result:
            addr = data.get("address") or {}
            # Accept the result unless it's only a state/country (too coarse); then try coarser zoom.
            is_coarse = result == addr.get("state") or result == addr.get("country")
            if not is_coarse:
                return result
    return result if result else None


# Matches DJI-style and common sensor filenames: DJI_YYYYMMDDHHMMSS_…
# Also matches bare YYYYMMDD_HHMMSS and YYYYMMDDHHMMSS patterns in filenames.
_FILENAME_DATETIME_RE = re.compile(
    r"(?:^|[_\-])(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])"  # YYYY MM DD
    r"(?:[_T]?(\d{2})(\d{2})(\d{2}))?",                           # optional HHMMSS
)


def _parse_filename_datetime(name: str) -> str | None:
    """Extract the earliest ISO 8601 datetime embedded in an image filename, or None."""
    m = _FILENAME_DATETIME_RE.search(name)
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), m.group(3)
    if m.group(4):
        return f"{year}-{month}-{day}T{m.group(4)}:{m.group(5)}:{m.group(6)}"
    return f"{year}-{month}-{day}"


def _temporal_hint_from_heads(
    file_heads: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Extract image-capture timestamps from the file inventory.

    Priority:
    1. ``images.json`` utc_time fields (Unix ms from EXIF — most authoritative).
    2. Filename datetime regex on image files present in file_heads.
    3. Filename datetime regex on ``filename`` fields inside images.json.

    Returns {"start": ISO, "end": ISO} or None.
    """
    from datetime import datetime, timezone

    # 1. images.json → utc_time (Unix ms, authoritative EXIF capture time)
    for head in file_heads:
        if str(head.get("name") or "").lower() == "images.json":
            path_str = str(head.get("path") or "")
            if not path_str:
                continue
            try:
                with open(path_str, encoding="utf-8") as fp:
                    records = json.load(fp)
                if not isinstance(records, list):
                    continue
                timestamps: list[str] = []
                for rec in records:
                    utc_ms = rec.get("utc_time")
                    if utc_ms is not None:
                        try:
                            dt = datetime.fromtimestamp(float(utc_ms) / 1000.0, tz=timezone.utc)
                            timestamps.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
                        except (ValueError, OverflowError, OSError):
                            pass
                if timestamps:
                    timestamps.sort()
                    result = {"start": timestamps[0], "end": timestamps[-1]}
                    logger.info("[temporal] extracted from images.json utc_time | start=%s end=%s", result["start"], result["end"])
                    return result
                # utc_time absent — fall through to filename parsing below
                for rec in records:
                    fn = str(rec.get("filename") or "")
                    dt_str = _parse_filename_datetime(fn)
                    if dt_str:
                        timestamps.append(dt_str)
                if timestamps:
                    timestamps.sort()
                    result = {"start": timestamps[0], "end": timestamps[-1]}
                    logger.info("[temporal] extracted from images.json filenames | start=%s end=%s", result["start"], result["end"])
                    return result
            except Exception:  # noqa: BLE001
                pass
            break

    # 2. Filename regex on image files in the file_heads inventory.
    image_exts = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".dng", ".raw"}
    dates: list[str] = []
    for head in file_heads:
        ext = str(head.get("extension") or "").lower()
        name = str(head.get("name") or "")
        if ext in image_exts:
            dt_str = _parse_filename_datetime(name)
            if dt_str:
                dates.append(dt_str)
    if dates:
        dates.sort()
        result = {"start": dates[0], "end": dates[-1]}
        logger.info("[temporal] extracted from image filenames | start=%s end=%s", result["start"], result["end"])
        return result

    return None


def _gather_evidence(request: dict[str, Any], settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the author's evidence: message/overrides/URLs + analyzed file reports.

    Returns ``(consolidated_inputs, file_inventory)``. File analysis routes each supplied
    file through the migrated extractors (text/tabular/json/geojson/pdf/image/raster/zip).
    Includes ``bbox_geojson_str`` and ``location_hint`` when a bounds file is found so the
    persona engine can give the author real geographic context for naming.
    """
    dataset = request.get("dataset") if isinstance(request.get("dataset"), dict) else {}
    source_urls = []
    if request.get("source_url"):
        source_urls.append(str(request["source_url"]))
    source_urls.extend(str(u) for u in request.get("source_urls") or [])

    evidence = gather_file_evidence(
        request, base_dir=settings.upload_root, deep_threshold=settings.deep_review_threshold
    )
    consolidated = {
        "user_message": str(request.get("message") or ""),
        "dataset_overrides": dataset,
        "source_urls": source_urls,
        "file_heads": evidence["file_heads"],
        "file_reports": evidence["file_reports"],
        "file_warnings": evidence["file_warnings"],
    }

    # Find a GeoJSON file with WGS84 coordinates to build the CKAN spatial field and
    # reverse-geocode a place name for the dataset title.
    spatial = _wgs84_spatial_from_heads(evidence["file_heads"])
    if spatial is not None:
        bbox_str, centroid = spatial
        consolidated["bbox_geojson_str"] = bbox_str
        hint = _reverse_geocode(centroid[0], centroid[1])
        if hint:
            consolidated["location_hint"] = hint

    # Extract temporal coverage from image filenames so the LLM gets authoritative
    # ISO 8601 dates rather than guessing from ambiguous filename formats.
    temporal = _temporal_hint_from_heads(evidence["file_heads"])
    if temporal:
        consolidated["temporal_hint"] = temporal

    return consolidated, build_file_inventory(evidence["file_heads"])


def _config_org_defaults(settings: Settings) -> dict[str, Any]:
    """Deployment-level values from configuration (.env/Settings).

    Includes structural/policy fields (org, license, CRS) and maintainer identity
    fields. Maintainer is always seeded from settings so the LLM never guesses it;
    _ground_user_profile overwrites maintainer with the live CKAN user when auth succeeds.
    Author is intentionally excluded — the persona asks the user for it.
    """
    pairs = [
        ("owner_org", settings.ckan_owner_org),
        ("license_id", settings.ckan_dataset_license_id),
        # Only seed contact email / CRS when explicitly configured at the deployment level.
        ("data_contact_email", settings.ckan_data_contact_email),
        ("coordinate_system", settings.ckan_coordinate_system),
        # Maintainer defaults — overwritten by the live CKAN user identity when auth succeeds.
        ("maintainer", settings.ckan_dataset_maintainer),
        ("maintainer_email", settings.ckan_dataset_maintainer_email),
    ]
    return {k: v for k, v in pairs if v}


def _authoritative_metadata(state: CkanRegistrationState, settings: Settings) -> dict[str, Any]:
    """Authoritative values, lowest→highest precedence:
    config org-defaults < llm-locked < thread-sticky org metadata < this dataset's clarifications.

    Maintainer is always seeded from settings/CKAN identity and is never cross-populated
    from author. Author is left for the persona to ask the user for.
    """
    merged: dict[str, Any] = _config_org_defaults(settings)
    merged.update(state.get("llm_locked_fields") or {})
    merged.update(state.get("org_metadata") or {})
    merged.update(state.get("dataset_clarifications") or {})
    return merged


def _effective_tapis_token(settings: Settings) -> str:
    """Per-request Tapis JWT (from the chat Authorization header) takes precedence over
    the static env-var token so the logged-in user's identity reaches the MCP server."""
    from app.auth_context import get_request_ckan_auth
    auth = get_request_ckan_auth() or ""
    if auth:
        # Strip "Bearer " scheme if present — MCP expects the bare token.
        return auth.removeprefix("Bearer ").removeprefix("bearer ").strip()
    return settings.mcp_tapis_token or ""


def _try_mcp_client(label: str, url: str, **kw: Any) -> Any:
    """Connect + ping an MCP server; return the client or None (graceful degradation)."""
    try:
        client = get_shared_client(url, **kw)
        if not client.ping():
            raise RuntimeError("server did not respond to ping")
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning("[persona] %s MCP unavailable (%s); skipping its tools", label, exc)
        return None


def _mcp_executor_and_schemas(
    settings: Settings, registry: ToolRegistry, allow: list[str]
) -> tuple[Any, list[dict[str, Any]]] | None:
    """Build a multi-server composite executor (CKAN + geo MCP + in-process) and merged schemas.

    Returns ``None`` when no MCP server is enabled/reachable (graceful fallback to in-process
    read tools, Fork B). Routing is a flat ``tool_name -> executor`` map across all servers.

    Safety: write/transform tools (``PERSONA_BLOCKED_TOOLS``) are excluded from the persona
    schema **unconditionally** (not just by allow-list) and hard-blocked in the executors; geo
    exposes only ``GEO_PERSONA_METADATA_TOOLS`` to personas (via a bounded submit-poll wrapper).
    Tokens are never model-visible: CKAN uses an HTTP header; geo metadata uses the server env or
    a server-side arg injection by the executor.
    """
    in_process_names = {s.name for s in registry.load_all()}
    all_names = set(in_process_names)
    mcp_tools: dict[str, Any] = {}
    in_process_allow = [n for n in allow if n in in_process_names]
    schemas: list[dict[str, Any]] = list(registry.to_openai_tools(names=in_process_allow))
    logger.info(
        "[persona_tools] in-process tools available=%s | allow-listed=%s | offered=%s",
        sorted(in_process_names), allow, in_process_allow,
    )

    # ── CKAN MCP server (arg-token injection, same pattern as geo) ───────────
    # The JWT is a Tapis token that expires every 6 h — it cannot live in the
    # shared transport headers. Inject it per-call via token_arg so every tool
    # invocation carries the current request's fresh token.
    if settings.mcp_enabled:
        logger.info("[persona_tools] CKAN MCP enabled — connecting to %s", settings.mcp_server_url)
        ckan = _try_mcp_client(
            "CKAN",
            settings.mcp_server_url,
            shared_secret=settings.mcp_shared_secret or None,
            timeout=settings.mcp_timeout,
        )
        if ckan is not None:
            ckan_names = set(ckan.tool_names())
            logger.info("[persona_tools] CKAN MCP connected | tools available: %s", sorted(ckan_names))
            _assert_no_overlap(all_names, ckan_names, "CKAN")
            all_names |= ckan_names
            ckan_exec = MCPToolExecutor(
                ckan,
                token_arg="tapis_token",
                token_value=_effective_tapis_token(settings) or None,
            )
            for n in ckan_names:
                mcp_tools[n] = ckan_exec
            ckan_allow = [n for n in allow if n in ckan_names and n not in PERSONA_BLOCKED_TOOLS]
            schemas += ckan.to_openai_tools(names=ckan_allow)
            logger.info("[persona_tools] CKAN tools offered to author: %s", ckan_allow)
        else:
            logger.warning("[persona_tools] CKAN MCP unreachable — no CKAN tools offered")
    else:
        logger.info("[persona_tools] CKAN MCP disabled (CKAN_MCP_ENABLED not set)")

    # ── Geo MCP server (arg-token injection; metadata via sync wrapper) ──────
    if settings.geo_mcp_enabled:
        logger.info("[persona_tools] geo MCP enabled — connecting to %s", settings.geo_mcp_url)
        geo = _try_mcp_client(
            "geo",
            settings.geo_mcp_url,
            shared_secret=settings.geo_mcp_shared_secret or None,
            timeout=settings.mcp_timeout,
        )
        if geo is not None:
            geo_names = set(geo.tool_names())
            logger.info("[persona_tools] geo MCP connected | tools available: %s", sorted(geo_names))
            _assert_no_overlap(all_names, geo_names, "geo")
            all_names |= geo_names
            token = _effective_tapis_token(settings) or settings.geo_mcp_tapis_token or None
            sync = GeoSyncExecutor(geo, token_value=token, poll_timeout=settings.geo_poll_timeout)
            plain = MCPToolExecutor(geo, token_arg="tapis_token", token_value=token)
            for n in geo_names:
                # metadata-extract → sync wrapper (submit+poll); everything else → plain
                # (transforms are hard-blocked inside the executor; only the gated node runs them)
                mcp_tools[n] = sync if n in GEO_PERSONA_METADATA_TOOLS else plain
            geo_allow = [n for n in allow if n in geo_names and n in GEO_PERSONA_METADATA_TOOLS]
            schemas += geo.to_openai_tools(names=geo_allow)
            logger.info("[persona_tools] geo tools offered to author: %s", geo_allow)
        else:
            logger.warning("[persona_tools] geo MCP unreachable — no geo tools offered")
    else:
        logger.info("[persona_tools] geo MCP disabled (GEO_MCP_ENABLED not set)")

    if not mcp_tools:
        logger.warning("[persona_tools] no MCP tools available — falling back to in-process only")
        return None
    executor = CompositeToolExecutor(InProcessToolExecutor(registry), mcp_tools)
    logger.info("[persona_tools] composite executor ready | total schemas offered: %d", len(schemas))
    return executor, schemas


def _assert_no_overlap(existing: set[str], new: set[str], label: str) -> None:
    overlap = existing & new
    if overlap:
        raise ToolError(f"{label} MCP tool names overlap existing tools; routing is ambiguous: {sorted(overlap)}")


def _tool_kwargs(settings: Settings, author: Persona) -> dict[str, Any]:
    """Build the engine's tool-calling kwargs when CKAN_PERSONA_TOOLS is on and the author
    persona declares a `tools:` allow-list; otherwise empty (no-tools path, unchanged).

    When CKAN_MCP_ENABLED is on and the server is reachable, CKAN tools are served by MCP and
    file tools stay in-process (composite). Otherwise the in-repo CKAN read tools are the
    fallback (Fork B). Write tools are never advertised to the author (Fork A): MCPToolExecutor
    hard-blocks live writes and the write-tool schemas are dry-run-only/token-scrubbed."""
    if not settings.persona_tools_enabled:
        logger.info("[persona_tools] CKAN_PERSONA_TOOLS not enabled — tool-calling skipped")
        return {}
    if not author.tools:
        logger.info("[persona_tools] author persona %r has no tools allow-list — tool-calling skipped", author.name)
        return {}
    logger.info("[persona_tools] author=%r allow-list=%s max_tool_calls=%d", author.name, list(author.tools), settings.max_tool_calls)
    registry = ToolRegistry(settings.tools_dir)
    allow = list(author.tools)

    composite = _mcp_executor_and_schemas(settings, registry, allow)
    if composite is not None:
        executor, author_tool_specs = composite
    else:
        executor = InProcessToolExecutor(registry)
        author_tool_specs = registry.to_openai_tools(names=allow)
    if not author_tool_specs:
        logger.warning("[persona_tools] no tool schemas resolved — tool-calling unavailable for this run")
        return {}

    def tool_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        return llm.invoke_chat_tools(
            messages,
            tools,
            model=settings.ckan_llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "",
        )

    return {
        "tool_executor": executor,
        "author_tool_specs": author_tool_specs,
        "tool_chat_fn": tool_chat,
        "max_tool_calls": settings.max_tool_calls,
    }


def _reply_text(resume_payload: Any) -> str:
    if isinstance(resume_payload, dict):
        return str(resume_payload.get("message") or resume_payload.get("answer") or resume_payload.get("schema") or "")
    return str(resume_payload or "")


def _match_profile(reply: Any, profiles: list[dict[str, Any]], default: str) -> str:
    """Map a free-text reply to a schema profile name (exact, substring, then when_to_use keyword)."""
    text = _reply_text(reply).strip().lower()
    names = [str(p.get("name", "")) for p in profiles]
    if not text:
        return default
    for name in names:
        if text == name.lower():
            return name
    for name in names:
        if name.lower() in text or text in name.lower():
            return name
    words = {w for w in re.split(r"\W+", text) if len(w) > 3}
    for profile in profiles:
        when = str(profile.get("when_to_use", "")).lower()
        if any(w in when for w in words):
            return str(profile.get("name"))
    return default


def _llm_classify_schema(
    request: dict[str, Any],
    profiles: list[dict[str, Any]],
    settings: Settings,
) -> str | None:
    """Auto-select a schema using a lightweight LLM call on file names + profile descriptions.

    Returns the best profile name, or None when the match is genuinely ambiguous.
    """
    if len(profiles) <= 1:
        return profiles[0]["name"] if profiles else None

    upload_dir = str(request.get("upload_dir") or "")
    file_names: list[str] = []
    if upload_dir:
        try:
            p = Path(upload_dir)
            if p.is_dir():
                file_names = [f.name for f in sorted(p.iterdir()) if f.is_file()][:30]
        except Exception:  # noqa: BLE001
            pass

    profile_block = "\n".join(
        f"- {p['name']}: {p.get('when_to_use', '').strip()}" for p in profiles
    )
    file_list = ", ".join(file_names) if file_names else "(no files)"
    user_message = str(request.get("message") or "").strip()

    prompt_parts = [
        "Select the best CKAN metadata schema for a dataset upload. "
        "Reply with ONLY the schema name exactly as written below, or reply 'ambiguous' if unclear.",
        "",
        f"Available schemas:\n{profile_block}",
        "",
        f"Uploaded files: {file_list}",
    ]
    if user_message:
        prompt_parts.append(f"User description: {user_message}")
    prompt_parts.append("\nSchema name (or 'ambiguous'):")

    try:
        text = llm.invoke_chat(
            [{"role": "user", "content": "\n".join(prompt_parts)}],
            model=settings.ckan_llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or "",
            max_tokens=32,
        ).strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[schema_select] LLM schema classification failed: %s", exc)
        return None

    if not text or "ambiguous" in text:
        return None
    names = [str(p.get("name", "")) for p in profiles]
    for name in names:
        if text == name.lower() or name.lower() in text:
            return name
    return None


def _select_schema_name(settings: Settings, state: CkanRegistrationState, request: dict[str, Any]) -> str:
    explicit = request.get("schema") or state.get("schema_profile")
    if explicit:
        return str(explicit)
    profiles = SchemaRegistry(settings.schemas_dir).list_profiles()
    if not settings.ask_schema or len(profiles) <= 1:
        return profiles[0]["name"] if len(profiles) == 1 else settings.default_schema_profile

    # Try LLM auto-classification before interrupting the user.
    auto = _llm_classify_schema(request, profiles, settings)
    if auto:
        logger.info("[schema_select] LLM auto-selected schema %r", auto)
        return auto

    # Only interrupt when the LLM couldn't confidently decide.
    lines = ["Which schema profile should I use for this dataset? Reply with the name.", ""]
    lines += [f"- **{p['name']}**: {p['when_to_use']}" for p in profiles]
    reply = interrupt(
        {
            "type": "schema_selection_required",
            "message": "\n".join(lines),
            "options": [p["name"] for p in profiles],
            "thread_id": state.get("thread_id"),
        }
    )
    return _match_profile(reply, profiles, settings.default_schema_profile)


def _ground_owner_org_field(settings: Settings, state: CkanRegistrationState, org_meta: dict[str, Any]) -> None:
    """Resolve owner_org against the portal's real orgs, mutating ``org_meta`` in place.

    configured-exists → canonical name; configured-absent + single org → that org;
    configured-absent + multiple → interrupt() to choose; portal unreachable/no orgs → leave
    the configured default (best-effort, never blocks). Respects an already-set owner_org.
    """
    from app.agents.ckan_registration.org_grounding import fetch_orgs, resolve_owner_org_choice

    if org_meta.get("owner_org"):
        return
    try:
        orgs = fetch_orgs(settings)
    except Exception as exc:  # noqa: BLE001 - grounding is best-effort
        logger.warning("[schema_select] could not fetch CKAN orgs (%s); using configured owner_org", exc)
        return

    resolved, ambiguous, options = resolve_owner_org_choice(orgs, settings.ckan_owner_org)
    if ambiguous:
        option_names = {o["name"] for o in options}
        option_lines = [f"- **{o['name']}**" + (f" — {o['title']}" if o["title"] else "") for o in options]
        preamble = ""
        while True:
            lines: list[str] = []
            if preamble:
                lines += [preamble, ""]
            lines += ["Which CKAN organization should own this dataset? Reply with the name.", ""]
            lines += option_lines
            reply = interrupt(
                {
                    "type": "owner_org_selection_required",
                    "message": "\n".join(lines),
                    "options": sorted(option_names),
                    "thread_id": state.get("thread_id"),
                }
            )
            reply_text = _reply_text(reply)
            if _is_meta_question(reply_text):
                preamble = "Here are your available organizations:"
                continue
            chosen = _match_owner_org(reply_text, option_names, settings.ckan_owner_org)
            if chosen in option_names:
                org_meta["owner_org"] = chosen
                break
            preamble = f"'{reply_text[:50]}' didn't match — please choose from the list below."
    elif resolved and resolved != settings.ckan_owner_org:
        logger.info("[schema_select] grounded owner_org %r → %r (live CKAN org)", settings.ckan_owner_org, resolved)
        org_meta["owner_org"] = resolved


def _ground_license_field(settings: Settings, org_meta: dict[str, Any]) -> None:
    """Resolve license_id against the portal's enabled licenses, mutating ``org_meta`` in place.

    Maps the configured license to the portal's canonical id when it exists there; leaves the
    configured value untouched otherwise (licenses have no safe fallback, so no guess/interrupt).
    """
    from app.agents.ckan_registration.org_grounding import fetch_licenses, resolve_license_id

    if org_meta.get("license_id"):
        return
    try:
        licenses = fetch_licenses(settings)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.warning("[schema_select] could not fetch CKAN licenses (%s); using configured license_id", exc)
        return
    resolved = resolve_license_id(licenses, settings.ckan_dataset_license_id)
    if resolved and resolved != settings.ckan_dataset_license_id:
        logger.info(
            "[schema_select] grounded license_id %r → %r (portal license)",
            settings.ckan_dataset_license_id,
            resolved,
        )
        org_meta["license_id"] = resolved


def _ground_user_profile(settings: Settings, org_meta: dict[str, Any]) -> None:
    """Pre-populate author and maintainer from the authenticated CKAN/Tapis user profile.

    Queries ``user_show?id=current`` with the per-request auth header (set by the TACC login).
    Maintainer is always overwritten with the logged-in identity. Author is only seeded when
    not already set (thread-sticky from a previous dataset or user-supplied).
    Degrades silently on any error (unauthenticated, old CKAN, network failure).
    """
    try:
        from app.agents.ckan_registration.auth import build_ckan_authorization_header
        auth = build_ckan_authorization_header(settings, required=False) or None
    except Exception:  # noqa: BLE001
        auth = None
    if not auth:
        return
    try:
        user = CkanClient(
            base_url=settings.ckan_url,
            authorization_header=auth,
            timeout=10,
        ).user_show_current()
        if not isinstance(user, dict):
            return
        name = str(user.get("display_name") or user.get("fullname") or user.get("name") or "").strip()
        email = str(user.get("email") or "").strip()
        if name:
            org_meta["maintainer"] = name
        if email and _valid_email(email):
            org_meta["maintainer_email"] = email
        if name or email:
            logger.info("[schema_select] pre-populated maintainer from CKAN user profile (%s)", name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[schema_select] could not fetch CKAN user profile: %s", exc)


def _match_owner_org(reply_text: str, names: set[str], default: str) -> str:
    text = str(reply_text or "").strip()
    lowered = text.casefold()
    for name in names:
        if name.casefold() == lowered:
            return name
    for name in names:
        if name.casefold() in lowered:
            return name
    return text or default


def make_schema_select_node(settings: Settings) -> Callable:
    def schema_select(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("schema_select", state, reason="Pick the CKAN schema profile + ground org/license")
        request = dict(state.get("request") or {})
        name = _select_schema_name(settings, state, request)
        update: dict[str, Any] = {"schema_profile": name}

        # Ground org-level values against the live portal (best-effort) into one org_metadata.
        before = dict(state.get("org_metadata") or {})
        org_meta = dict(before)
        # Seed owner_org from the caller's dataset override so a known org (e.g. on
        # re-publish) bypasses the interrupt without asking the user again.
        if not org_meta.get("owner_org"):
            _ds_override = request.get("dataset") if isinstance(request.get("dataset"), dict) else {}
            _pre_org = str(_ds_override.get("owner_org") or "").strip()
            if _pre_org:
                org_meta["owner_org"] = _pre_org
        _ground_owner_org_field(settings, state, org_meta)
        _ground_license_field(settings, org_meta)
        # Pre-populate contact fields from the authenticated CKAN/TACC user profile so the
        # agent never needs to ask the uploader for their own name and email.
        _ground_user_profile(settings, org_meta)
        if org_meta != before:
            update["org_metadata"] = org_meta

        log_node_exit("schema_select", {"schema_profile": name}, next_node="persona")
        return update

    return schema_select


def make_persona_node(settings: Settings, *, engine: EngineFn = run_persona_metadata_loop) -> Callable:
    def persona(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("persona", state, reason="Persona author drafts + curator/scientist review")
        request = dict(state.get("request") or {})
        schema_name = str(request.get("schema") or state.get("schema_profile") or settings.default_schema_profile)
        profile = SchemaRegistry(settings.schemas_dir).get(schema_name)
        registry = PersonaRegistry(settings.personas_dir)
        author = registry.author()
        consolidated, file_inventory = _gather_evidence(request, settings)
        # Extract bbox from consolidated so it goes to the engine separately (not nested under
        # consolidated_inputs) — the engine includes it as a top-level payload key.
        bbox_geojson = consolidated.pop("bbox_geojson_str", None)

        # When the caller sends a resume message (user revision), surface it to the author
        # and release the title lock so the user's instruction can override the prior title.
        resume = request.get("resume") if isinstance(request.get("resume"), dict) else {}
        resume_msg = str(resume.get("message") or "").strip()
        is_resume = bool(resume_msg)
        if is_resume:
            existing_msg = consolidated.get("user_message", "")
            consolidated["user_message"] = (
                f"{existing_msg}\n\nUser revision request: {resume_msg}".strip()
            )

        # Build authoritative metadata; on a resume drop the locked title so the author
        # re-derives it from the user's instruction rather than repeating the old value.
        auth_meta = dict(_authoritative_metadata(state, settings))
        if is_resume:
            auth_meta.pop("title", None)

        tool_kwargs = _tool_kwargs(settings, author)

        result = engine(
            consolidated,
            author_persona=author,
            evaluator_personas=registry.evaluators(),
            schema_profile=profile,
            file_inventory=file_inventory or None,
            bbox_geojson=bbox_geojson,
            organizational_metadata=auth_meta or None,
            llm_model=settings.ckan_llm_model,
            llm_api_key=settings.openai_api_key,
            llm_base_url=settings.openai_base_url,
            model_id=str(request.get("session_id") or state.get("thread_id") or ""),
            runs_dir=settings.runs_dir,
            **tool_kwargs,
        )
        next_node = "clarify" if result.stop_reason == STOP_NEEDS_CLARIFICATION else "propose"
        output: dict[str, Any] = {
            "schema_profile": schema_name,
            "candidate_metadata": result.proposed_metadata,
            "evaluator_verdicts": [
                {"persona": v.persona_name, "verdict": v.verdict, "questions": v.questions}
                for r in result.transcript
                for v in r.evaluator_verdicts
            ],
            "persona_stop_reason": result.stop_reason,
            "clarification_questions": result.clarification_questions,
            "reviewed_files": [h.get("name") for h in consolidated.get("file_heads", []) if h.get("name")],
            "status": result.stop_reason,
            # Clear any stale result carried over from a prior run on this thread
            # (e.g. a single-pass metadata report) so the response reflects only this run.
            "result": None,
        }

        # Lock the title after the first successful draft so internal persona re-runs don't
        # drift to different phrasing. On a resume the old lock was already released above,
        # so the new candidate title replaces it here regardless of prior lock state.
        proposed = result.proposed_metadata or {}
        candidate_title = str(proposed.get("title") or "").strip()
        title_lockable = (
            candidate_title
            and "untitled" not in candidate_title.lower()
            and not candidate_title.startswith("_gap_")
            and not re.match(r"^Task of \d{4}-\d{2}-\d{2}T", candidate_title)
        )
        if title_lockable and (
            is_resume or "title" not in (state.get("llm_locked_fields") or {})
        ):
            output["llm_locked_fields"] = {
                **(state.get("llm_locked_fields") or {}),
                "title": candidate_title,
            }

        log_node_exit("persona", {"status": result.stop_reason}, next_node=next_node)
        return output

    return persona


def _clarify_message(current: dict[str, Any], state: CkanRegistrationState, questions: list[dict[str, Any]]) -> str:
    """Build the interrupt message with dataset context + reason so the user understands why they're being asked."""
    parts: list[str] = []

    # Dataset title — prefer the locked version over the latest candidate draft
    candidate = state.get("candidate_metadata") or {}
    title = str(
        (state.get("llm_locked_fields") or {}).get("title")
        or (state.get("dataset_clarifications") or {}).get("title")
        or candidate.get("title") or ""
    ).strip()
    if title:
        parts.append(f"**Dataset:** {title}")

    # Validation error from the previous round for this field (e.g. invalid email).
    field = str(current.get("field") or "")
    prior_error = (state.get("clarification_errors") or {}).get(field, "")
    if prior_error:
        parts.append(f"⚠️ {prior_error} — please try again.")

    # The evaluator's question text
    question_text = (current.get("question") or "The agent needs more information.").strip()
    parts.append(question_text)

    # Why the evaluator couldn't derive it from the source files
    reason = str(current.get("reason_not_derivable") or "").strip()
    if reason:
        parts.append(f"_{reason}_")

    # Field-specific context and shortcuts
    ctx = _FIELD_CONTEXT.get(field, "")
    # For email fields, dynamically inject the known author email as a shortcut hint.
    if not ctx and field in {"data_contact_email", "maintainer_email"}:
        known = {**(state.get("org_metadata") or {}), **(state.get("dataset_clarifications") or {})}
        author_email = known.get("author_email", "")
        if author_email:
            ctx = f"Author email on file: `{author_email}` — type `same as author` to reuse it."
    if ctx:
        parts.append(ctx)

    remaining = len(questions) - 1
    if remaining > 0:
        parts.append(f"({remaining} more question{'s' if remaining > 1 else ''} after this)")

    return "\n\n".join(parts)


def make_clarify_node() -> Callable:
    def clarify(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("clarify", state, reason="Interrupt: ask user for one non-derivable field")

        # ── Skip guard ────────────────────────────────────────────────────────
        # Drop questions for fields the user already answered or explicitly declined.
        # This prevents the persona LLM from re-asking a field on subsequent rounds
        # even after the user provided a valid answer or said "no".
        # Build effective user/LLM-provided values (config defaults excluded — those are
        # never the source of a user answer so need no skip check).
        answered: dict[str, Any] = {}
        for src in (
            state.get("llm_locked_fields") or {},
            state.get("org_metadata") or {},
            state.get("dataset_clarifications") or {},
        ):
            if isinstance(src, dict):
                answered.update(src)

        declined = set(state.get("declined_fields") or [])
        pending: list[dict[str, Any]] = []
        for q in (state.get("clarification_questions") or []):
            field = q.get("field", "")
            if field in declined:
                logger.info("[clarify] field %r was declined by user; skipping re-ask", field)
                continue
            existing = answered.get(field)
            if field in EMAIL_FIELDS:
                if _valid_email(str(existing or "")):
                    logger.info("[clarify] field %r already has a valid email in state; skipping", field)
                    continue
            elif existing is not None and str(existing).strip():
                logger.info("[clarify] field %r already has a value in state; skipping", field)
                continue
            pending.append(q)

        if not pending:
            # All questions already answered — return early without interrupting.
            return {"clarification_round": int(state.get("clarification_round") or 0) + 1}

        # Sort by defined presentation order so the conversation flows logically
        # (author → author_email → maintainer → maintainer_email → …).
        pending.sort(key=lambda q: _FIELD_PRIORITY.get(q.get("field", ""), 99))
        questions = pending
        current = questions[0]
        if not current.get("field"):
            current = {**current, "field": _slugify(current.get("question") or "", "clarification")}

        # ── Single interrupt per clarify round ───────────────────────────────
        resume_payload = interrupt({
            "type": "metadata_clarification_required",
            "message": _clarify_message(current, state, questions),
            "field": current.get("field"),
            "questions": [current],
            "remaining": len(questions),
            "thread_id": state.get("thread_id"),
        })
        answers = _parse_clarification_answers(resume_payload, [current])

        # If the user typed a question or confusion signal instead of a value, re-interrupt
        # with field guidance rather than storing the meta-text (e.g. "what are my options?").
        field_val = answers.get(str(current.get("field") or ""))
        if field_val and _is_meta_question(str(field_val)):
            guidance = _field_guidance_message(current, state, questions)
            resume_payload = interrupt({
                "type": "metadata_clarification_required",
                "message": guidance,
                "field": current.get("field"),
                "questions": [current],
                "remaining": len(questions),
                "thread_id": state.get("thread_id"),
            })
            answers = _parse_clarification_answers(resume_payload, [current])

        # ── Email field handling ──────────────────────────────────────────────
        # "no" / "n/a" / "skip" → record the decline and skip storage.
        # Invalid format → store error in clarification_errors for the next round's message.
        # Both cases avoid a second interrupt() call (which causes LangGraph replay issues).
        new_declined = list(state.get("declined_fields") or [])
        new_errors: dict[str, str] = dict(state.get("clarification_errors") or {})
        answered_fields: set[str] = set()

        for key in [k for k, v in list(answers.items()) if k in EMAIL_FIELDS]:
            raw = str(answers[key])

            # Explicit decline ("no", "n/a", "skip", …)
            if _NO_VALUE_RE.match(raw):
                logger.info("[clarify] user declined to provide %s (%r); marking declined", key, raw)
                answers.pop(key)
                if key not in new_declined:
                    new_declined.append(key)
                new_errors.pop(key, None)
                continue

            # Resolve "same as author / same as maintainer" shorthands
            resolved = _resolve_field_reference(raw, state)

            # Invalid email format: store error, don't store the field.
            # The next clarify round will show the error and re-ask.
            if not _valid_email(resolved):
                logger.warning("[clarify] invalid email for %s (%r); storing error for next round", key, resolved)
                answers.pop(key)
                if _SAME_AS_AUTHOR_RE.search(raw) or _SAME_AS_MAINTAINER_RE.search(raw):
                    new_errors[key] = (
                        "I couldn't find a stored email to reuse — "
                        "please enter the address directly (e.g. `name@example.org`)"
                    )
                else:
                    new_errors[key] = (
                        f"`{resolved}` isn't a valid email address — "
                        "please include the full domain (e.g. `name@example.org`)"
                    )
                continue

            # Valid email: store it and clear any prior error for this field.
            answers[key] = resolved
            new_errors.pop(key, None)
            answered_fields.add(key)

        # Track non-email answered fields so we can also clear their errors.
        answered_fields.update(k for k in answers if k not in EMAIL_FIELDS)
        for k in answered_fields:
            new_errors.pop(k, None)

        org_update = {k: v for k, v in answers.items() if k in ORG_LEVEL_FIELDS}
        dataset_update = {k: v for k, v in answers.items() if k not in ORG_LEVEL_FIELDS}

        output: dict[str, Any] = {
            "org_metadata": {**(state.get("org_metadata") or {}), **org_update},
            "dataset_clarifications": {**(state.get("dataset_clarifications") or {}), **dataset_update},
            "clarification_round": int(state.get("clarification_round") or 0) + 1,
            "clarification_errors": new_errors,
            "declined_fields": new_declined,
        }
        log_node_exit("clarify", {"answered": sorted(answers), "declined": sorted(new_declined), "errors": sorted(new_errors)}, next_node="persona")
        return output

    return clarify


_SAME_AS_AUTHOR_RE = re.compile(
    r"\b(same\s+as\s+author|author.s?\s+email|use\s+author|author\s+email)\b", re.I
)
_SAME_AS_MAINTAINER_RE = re.compile(
    r"\b(same\s+as\s+maintainer|maintainer.s?\s+email|use\s+maintainer)\b", re.I
)
# "no", "n/a", "none", "skip" for email fields → the field should be left null.
_NO_VALUE_RE = re.compile(
    r"^\s*(no|n/?a|none|skip|not\s+available|no\s+email|not\s+applicable|leave\s+blank|blank)\s*$",
    re.I,
)


def _resolve_field_reference(answer: str, state: CkanRegistrationState) -> str:
    """Resolve natural-language references like 'same as author' to the actual stored value.

    Only returns a resolved value when it is a syntactically valid email — prevents a person's
    name from silently substituting for an email address when author_email is not yet in state.
    Checks all state buckets (org_metadata, dataset_clarifications, llm_locked_fields) so
    inline-retry values that landed in any slot are still found.
    """
    text = answer.strip()
    known: dict[str, str] = {}
    for src in (
        state.get("org_metadata") or {},
        state.get("dataset_clarifications") or {},
        state.get("llm_locked_fields") or {},
    ):
        if isinstance(src, dict):
            known.update({k: str(v) for k, v in src.items() if v})

    if _SAME_AS_AUTHOR_RE.search(text):
        candidate = known.get("author_email") or known.get("author") or ""
        # Only substitute if the resolved value is actually a valid email; otherwise keep
        # the original text so the email-validation path can fire an informative error.
        return candidate if _valid_email(str(candidate)) else answer
    if _SAME_AS_MAINTAINER_RE.search(text):
        candidate = known.get("maintainer_email") or known.get("maintainer") or ""
        return candidate if _valid_email(str(candidate)) else answer
    return answer


def _parse_clarification_answers(resume_payload: Any, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Accept ``{"clarifications": {field: value}}`` or a single free-text answer.

    A bare string / ``{"message": ...}`` answer is attached to the first question's
    ``field`` when there is exactly one outstanding question.
    """
    if isinstance(resume_payload, dict):
        clar = resume_payload.get("clarifications")
        if isinstance(clar, dict) and clar:
            return {str(k): v for k, v in clar.items()}
        text = resume_payload.get("message") or resume_payload.get("answer")
    else:
        text = resume_payload

    text = str(text or "").strip()
    if text and len(questions) == 1 and questions[0].get("field"):
        return {str(questions[0]["field"]): text}
    return {}


def _controlled_vocab_violations(payload: dict[str, Any], controlled_vocab: dict[str, Any]) -> dict[str, Any]:
    """Return select-field values in ``payload`` that fall outside the schema's controlled vocab.

    ``controlled_vocab`` maps a field name to its list of allowed values. Returns
    ``{field: {"invalid": [...], "allowed": [...]}}`` for each violating field (empty when clean).
    Comparison is case-insensitive; list-valued fields (e.g. ``categories``) are checked per item.
    """
    violations: dict[str, Any] = {}
    for field_name, allowed in (controlled_vocab or {}).items():
        if not isinstance(allowed, list) or not allowed:
            continue
        value = payload.get(field_name)
        if value in (None, "", []):
            continue
        allowed_norm = {str(a).strip().casefold() for a in allowed}
        values = value if isinstance(value, list) else [value]
        invalid = [v for v in values if str(v).strip().casefold() not in allowed_norm]
        if invalid:
            violations[field_name] = {"invalid": invalid, "allowed": allowed}
    return violations


def _search_existing_datasets(settings: Settings, desired: dict[str, Any]) -> list[dict[str, Any]]:
    """Search CKAN for datasets with similar title/name. Returns up to 5 scored candidates."""
    title = str(desired.get("title") or desired.get("name") or "").strip()
    if not title:
        return []
    try:
        results = CkanClient(base_url=settings.ckan_url, timeout=15).package_search(title, rows=10)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[propose] CKAN existing-dataset search failed: %s", exc)
        return []
    candidates = []
    for dataset in results:
        if not isinstance(dataset, dict):
            continue
        haystack = " ".join(str(dataset.get(k) or "") for k in ("name", "title")).lower()
        query_lower = title.lower()
        score = difflib.SequenceMatcher(None, query_lower, haystack[: max(len(query_lower) * 3, 120)]).ratio()
        if query_lower in haystack:
            score = min(score + 0.25, 1.0)
        if score < 0.25:
            continue
        candidates.append({
            "id": str(dataset.get("id") or ""),
            "name": str(dataset.get("name") or ""),
            "title": str(dataset.get("title") or ""),
            "score": round(score, 3),
        })
    return sorted(candidates, key=lambda c: -c["score"])[:5]


def make_propose_node(settings: Settings) -> Callable:
    def propose(state: CkanRegistrationState) -> dict[str, Any]:
        log_node_entry("propose", state, reason="Emit labeled proposal + write analyzed state")
        request = dict(state.get("request") or {})
        candidate = dict(state.get("candidate_metadata") or {})
        schema_name = str(state.get("schema_profile") or settings.default_schema_profile)
        profile = SchemaRegistry(settings.schemas_dir).get(schema_name)

        dataset_overrides = request.get("dataset") if isinstance(request.get("dataset"), dict) else {}
        user_keys = set(dataset_overrides) | set(state.get("org_metadata") or {}) | set(
            state.get("dataset_clarifications") or {}
        )
        # Build the full proposed payload (every schema field the author populated, plus hard
        # defaults), then label origins against that payload so the review shows them all.
        desired = _to_desired_payload(candidate, settings, profile)
        origins = _field_origins(desired, profile.defaults, user_keys)

        capped = int(state.get("clarification_round") or 0) >= CLARIFICATION_CAP
        outstanding = state.get("clarification_questions") or []
        reviewed_files = state.get("reviewed_files") or []
        # The drafting loop may have stopped short (a rate-limit/LLM error, or running out
        # of refinement rounds) — in which case fields like `notes` can be a pre-improvement
        # stub. Surface that rather than presenting the draft as final.
        stop_reason = str(state.get("persona_stop_reason") or "")
        degraded = stop_reason in {STOP_LLM_ERROR, STOP_MAX_ROUNDS}

        # Build resource plan from uploaded files so the apply step has paths to upload.
        # Also append any pre-specified remote URL resources (e.g. WebODM outputs).
        evidence, _ = _gather_evidence(request, settings)
        resource_plan = _resource_plan_from_heads(evidence.get("file_heads") or [])
        for _item in (request.get("remote_resources") or []):
            _url = str((_item.get("url") if isinstance(_item, dict) else _item) or "").strip()
            if not _url:
                continue
            _name = str(_item.get("name") or "") if isinstance(_item, dict) else ""
            _fmt = str(_item.get("format") or "") if isinstance(_item, dict) else ""
            _desc = str(_item.get("description") or "") if isinstance(_item, dict) else ""
            _bare = _url.split("?")[0]
            if not _name:
                _name = Path(_bare).name or _url
            if not _fmt:
                _fmt = Path(_bare).suffix.lstrip(".").upper() or "URL"
            resource_plan.append({
                "resource_name": re.sub(r"[^a-z0-9]+", "-", Path(_name).stem.lower()).strip("-") or "resource",
                "resource_title": Path(_name).stem.replace("-", " ").replace("_", " ").title() or "Resource",
                "resource_description": _desc or f"Remote asset: {_name}",
                "resource_url": _url,
                "format": _fmt,
                "mimetype": mimetypes.guess_type(_bare)[0] or "application/octet-stream",
            })

        session_id = _sanitize_session(str(request.get("session_id") or state.get("thread_id") or ""))
        state_path = _write_analyzed_state(
            settings, session_id, desired, origins, schema_name,
            resource_plan=resource_plan,
        )

        # Search for similar existing CKAN datasets so the review can surface them.
        existing_candidates: list[dict[str, Any]] = []
        try:
            existing_candidates = _search_existing_datasets(settings, desired)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[propose] existing dataset search failed: %s", exc)

        # Validate the proposal against CKAN via MCP immediately so the user can reply
        # `REGISTER` without needing a separate explicit dry-run step.
        mcp_validation: dict[str, Any] | None = None
        if settings.mcp_enabled:
            ckan_client = _try_mcp_client(
                "CKAN",
                settings.mcp_server_url,
                shared_secret=settings.mcp_shared_secret or None,
                tapis_token=_effective_tapis_token(settings) or None,
                timeout=settings.mcp_timeout,
            )
            if ckan_client is not None:
                _INTERNAL = frozenset({"owner_org_label", "owner_org_name", "owner_org_title", "isopen"})
                metadata_payload = {
                    k: v for k, v in desired.items()
                    if k not in _INTERNAL and v not in (None, "", [], {})
                }
                dataset_type = str(metadata_payload.get("type") or settings.ckan_dataset_type or "dataset")
                try:
                    pkg_result = ckan_client.call_tool("schema_create_package", {
                        "dataset_type": dataset_type,
                        "metadata": metadata_payload,
                        "dry_run": True,
                    })
                    valid = bool(pkg_result.get("valid")) if isinstance(pkg_result, dict) else False
                    errors = list(pkg_result.get("errors") or []) if isinstance(pkg_result, dict) else []
                    warnings_out = list(pkg_result.get("warnings") or []) if isinstance(pkg_result, dict) else []
                    mcp_validation = {"valid": valid, "errors": errors, "warnings": warnings_out}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[propose] MCP validation skipped: %s", exc)

        # Consolidate state updates: status, registration_intent, candidates, dry_run_result.
        state_updates: dict[str, Any] = {}
        if mcp_validation and mcp_validation.get("valid"):
            state_updates["status"] = "dry_run"
            state_updates["dry_run_result"] = mcp_validation
        if existing_candidates:
            state_updates["candidate_existing_datasets"] = existing_candidates
        # When no strong existing match, pre-resolve intent to "new" so the dry-run
        # approval interrupt is skipped when the user explicitly requests a validation step.
        if not any(c["score"] >= 0.75 for c in existing_candidates):
            state_updates["registration_intent"] = "new"
        if state_updates:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            saved.update(state_updates)
            tmp = state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(state_path)

        review_markdown = _review_markdown(
            desired, origins, capped, outstanding, reviewed_files,
            stop_reason=stop_reason, profile=profile, candidate=candidate,
            mcp_validation=mcp_validation, existing_candidates=existing_candidates or None,
        )

        # Enforce the schema's controlled vocabulary on the drafted values: flag any select-field
        # value the author produced that isn't an allowed choice (surfaced, not silently changed).
        vocab_violations = _controlled_vocab_violations(desired, profile.controlled_vocab)
        if vocab_violations:
            vlines = ["", "### ⚠️ Values outside the schema's controlled vocabulary"]
            for field_name, info in vocab_violations.items():
                vlines.append(
                    f"- **{field_name}**: {info['invalid']} — allowed: {info['allowed']}"
                )
            review_markdown = review_markdown + "\n".join(vlines)

        final_status = "dry_run" if mcp_validation and mcp_validation.get("valid") else "analyzed"
        result = {
            "ok": True,
            "command": "analyze",
            "status": final_status,
            "session_id": session_id,
            "state_path": str(state_path),
            "schema_profile": schema_name,
            "desired_dataset_payload": desired,
            "field_origins": origins,
            "review_markdown": review_markdown,
        }
        if mcp_validation is not None:
            result["mcp_validation"] = mcp_validation
        if degraded:
            result["degraded"] = True
            result["persona_stop_reason"] = stop_reason
        if capped and outstanding:
            result["outstanding_clarifications"] = outstanding
        if vocab_violations:
            result["controlled_vocab_violations"] = vocab_violations

        # Persona-proposed geo transform (spec 2026-06-30): if the author emitted a structured
        # `_transform_proposal`, validate its shape and hand it to the gated geo-approval node.
        # The model only proposes — it never executes (transforms stay hard-blocked in the loop).
        proposal = candidate.get("_transform_proposal")
        output: dict[str, Any] = {"result": result, "status": final_status, "error": ""}
        if isinstance(proposal, dict) and proposal:
            from app.agents.ckan_registration.geo_transform import (
                TransformProposalError,
                build_tool_call,
            )

            try:
                build_tool_call(proposal)  # shape check only; raises on invalid
                output["transform_request"] = proposal
                result["transform_proposed"] = True
                log_node_exit("propose", {"status": final_status}, next_node="geo-approval")
                return output
            except TransformProposalError as exc:
                result["transform_proposal_error"] = str(exc)
        log_node_exit("propose", {"status": final_status}, next_node="END")
        return output

    return propose


def route_after_propose(state: CkanRegistrationState) -> str:
    """Route to the gated geo-approval node when the persona proposed a valid transform."""
    proposal = state.get("transform_request")
    return "geo-approval" if isinstance(proposal, dict) and proposal else "END"


def _field_origins(candidate: dict[str, Any], defaults: dict[str, Any], user_keys: set[str]) -> dict[str, str]:
    origins: dict[str, str] = {}
    for key, value in candidate.items():
        if key.startswith("_gap_") or value in (None, "", []):
            continue
        if key in user_keys:
            origins[key] = "user-supplied"
        elif key in defaults:
            origins[key] = "schema-default"
        else:
            origins[key] = "llm-derived"
    return origins


def _normalize_date(value: str) -> str:
    """Strip any time component from ISO datetimes, leaving a bare YYYY-MM-DD."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(value or ""))
    return m.group(1) if m else value


def _to_desired_payload(candidate: dict[str, Any], settings: Settings, profile: SchemaProfile) -> dict[str, Any]:
    def val(*keys: str) -> str:
        for k in keys:
            v = candidate.get(k)
            if v:
                return str(v)
        return ""

    tag_string = val("tag_string")
    if tag_string:
        tags = [{"name": _slugify(t, "")} for t in tag_string.split(",") if _slugify(t, "")]
    else:
        raw_tags = candidate.get("tags") or []
        if isinstance(raw_tags, list):
            tags = [{"name": _slugify(str(t), "")} for t in raw_tags if _slugify(str(t), "")]
        else:
            tags = []
    title = val("title") or "Untitled Dataset"
    desired: dict[str, Any] = {
        "name": _slugify(val("name") or title),
        "title": title,
        "notes": val("notes"),
        "url": val("url"),
        "owner_org": val("owner_org") or settings.ckan_owner_org,
        "author": val("author"),
        "author_email": val("author_email"),
        "maintainer": val("maintainer"),
        "maintainer_email": val("maintainer_email"),
        "license_id": val("license_id"),
        "version": val("version"),
        "spatial": val("spatial"),
        "temporal_coverage_start": _normalize_date(val("temporal_coverage_start", "from_date")),
        "temporal_coverage_end": _normalize_date(val("temporal_coverage_end", "to_date")),
        "tags": tags,
    }
    # Append survey date to name to prevent collisions when multiple surveys of the same
    # site are registered (e.g. three Bethel runs all generate "bethel-runs-orthophoto-3d-model").
    _tcs = (desired.get("temporal_coverage_start") or "").split("T")[0][:10]
    if _tcs and _tcs not in desired["name"]:
        desired["name"] = _slugify(desired["name"] + "-" + _tcs)
    # Carry every other field the schema defines that the author populated. The CKAN-core set
    # above was the only thing previously kept, so subside-specific fields (categories,
    # collection_method, coordinate_system, program_area, caveats_usage, …) were dropped — and
    # therefore neither shown for review nor registered. Preserve them verbatim.
    for spec in profile.fields:
        key = str(spec.get("key") or "").strip()
        if not key or key in desired:
            continue
        value = candidate.get(key)
        if value not in (None, "", []):
            desired[key] = value
    # Apply hard schema defaults for anything still missing (e.g. required categories /
    # collection_method for subside).
    for key, value in (profile.defaults or {}).items():
        if desired.get(key) in (None, "", []):
            desired[key] = value
    return desired


_DEGRADED_BANNER = {
    STOP_LLM_ERROR: (
        "⚠️ The drafting model stopped on an error (often a rate limit) before it finished "
        "refining. Some fields — especially `notes` — may be incomplete. Re-run to let the "
        "agent finish, or lower the call rate via `LLM_CALL_DELAY_SECONDS`."
    ),
    STOP_MAX_ROUNDS: (
        "⚠️ The reviewers did not fully converge within the round limit, so some fields "
        "(e.g. `notes`) may still be thin. Re-run or add detail before registering."
    ),
}


# Most-reviewed fields first; any remaining schema fields follow alphabetically.
_REVIEW_FIELD_ORDER = [
    "name", "title", "notes", "owner_org", "author", "author_email", "maintainer",
    "maintainer_email", "license_id", "url", "version", "tags", "categories",
    "collection_method", "spatial", "coordinate_system",
    "temporal_coverage_start", "temporal_coverage_end",
]


def _fmt_value(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(v.get("name") if isinstance(v, dict) else v).strip() for v in value]
        return ", ".join(p for p in parts if p) or "<none>"
    return str(value) if value not in (None, "") else "<none>"


def _fmt_spatial(value: str) -> str:
    """Format a GeoJSON string as a readable geometry summary."""
    try:
        geo = json.loads(value) if isinstance(value, str) else value
        gtype = geo.get("type", "Geometry")
        coords = geo.get("coordinates")
        if gtype == "Polygon" and coords:
            flat = [c for ring in coords for c in ring]
            lons, lats = [c[0] for c in flat], [c[1] for c in flat]
            return (f"`{gtype}` rectangle: "
                    f"W={min(lons):.5f} E={max(lons):.5f} "
                    f"S={min(lats):.5f} N={max(lats):.5f}")
        if gtype == "Point" and coords and len(coords) >= 2:
            return f"`{gtype}` ({coords[1]:.5f}°N, {coords[0]:.5f}°E)"
        return f"`{gtype}`"
    except Exception:  # noqa: BLE001
        raw = str(value)
        return raw[:120] + "…" if len(raw) > 120 else raw


def _review_markdown(
    desired: dict[str, Any],
    origins: dict[str, str],
    capped: bool,
    outstanding: list[dict[str, Any]],
    reviewed_files: list[str] | None = None,
    stop_reason: str = "",
    profile: SchemaProfile | None = None,
    candidate: dict[str, Any] | None = None,
    mcp_validation: dict[str, Any] | None = None,
    existing_candidates: list[dict[str, Any]] | None = None,
) -> str:
    lines = ["## Proposed CKAN Metadata", ""]
    banner = _DEGRADED_BANNER.get(stop_reason)
    if banner:
        lines += [banner, ""]
    if reviewed_files:
        shown = ", ".join(f"`{n}`" for n in reviewed_files[:12])
        more = "" if len(reviewed_files) <= 12 else f" (+{len(reviewed_files) - 12} more)"
        lines += [f"**Files reviewed ({len(reviewed_files)}):** {shown}{more}", ""]
    lines += [
        "Each field is labeled with its origin so you can review before registering "
        "(`user-supplied`, `llm-derived`, `schema-default`).",
        "",
    ]

    defaults = (profile.defaults if profile else {}) or {}
    populated = {k: v for k, v in desired.items() if v not in (None, "", [])}
    ordered = [k for k in _REVIEW_FIELD_ORDER if k in populated]
    ordered += sorted(k for k in populated if k not in _REVIEW_FIELD_ORDER)
    for key in ordered:
        origin = origins.get(key) or ("schema-default" if key in defaults else "llm-derived")
        formatted = _fmt_spatial(desired[key]) if key == "spatial" else _fmt_value(desired[key])
        lines.append(f"- **{key}** (`{origin}`): {formatted}")

    # Show the schema fields that are still empty, with the author's gap reason when present,
    # so the user can see what is missing rather than assuming the proposal is complete.
    gaps = {
        k[len("_gap_"):]: str(v)
        for k, v in (candidate or {}).items()
        if k.startswith("_gap_")
    }
    if profile:
        missing = []
        for spec in profile.fields:
            key = str(spec.get("key") or "").strip()
            if not key or key in populated:
                continue
            required = " (required)" if spec.get("required") else ""
            reason = f" — {gaps[key]}" if key in gaps else ""
            missing.append(f"- **{key}**{required}: _not set_{reason}")
        if missing:
            lines += ["", "### Not set / needs input", *missing]

    if capped and outstanding:
        lines += ["", "### Unresolved (clarification cap reached)"]
        lines += [f"- {q.get('question')}" for q in outstanding]

    if existing_candidates:
        lines += ["", "### Similar datasets already in CKAN"]
        lines += ["These packages share a similar name or title — confirm you want a new one, or tell me which to update:", ""]
        for c in existing_candidates:
            label = c.get("title") or c["name"]
            lines.append(f"- `{c['name']}` — {label} ({int(c['score'] * 100)}% match)")

    lines += [""]
    has_candidates = bool(existing_candidates)
    if mcp_validation is None:
        if has_candidates:
            lines.append("Reply `REGISTER` to create a new dataset, `update <name>` to update one above, or send corrections.")
        else:
            lines.append("Reply `REGISTER` when ready, or send corrections to revise any field.")
    else:
        errors = mcp_validation.get("errors") or []
        warnings_list = mcp_validation.get("warnings") or []
        if errors:
            lines += ["### Validation errors"]
            lines += [f"- {e}" for e in errors]
        if warnings_list:
            lines += ["### Validation warnings"]
            lines += [f"- {w}" for w in warnings_list]
        if mcp_validation.get("valid"):
            if has_candidates:
                lines.append("Metadata validated ✓ — reply `REGISTER` to create new, `update <name>` to update one above, or tell me what to change.")
            else:
                lines.append("Metadata validated ✓ — reply `REGISTER` to create this dataset, or tell me what to change.")
        else:
            lines.append("Some fields need attention — correct them and I'll re-validate, or send `REGISTER` to attempt registration anyway.")

    return "\n".join(lines)


def _sanitize_session(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return text or uuid.uuid4().hex


def _write_analyzed_state(
    settings: Settings,
    session_id: str,
    desired: dict[str, Any],
    origins: dict[str, str],
    schema_name: str,
    *,
    resource_plan: list[dict[str, Any]] | None = None,
    status: str = "analyzed",
) -> Path:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    path = settings.state_dir / f"{session_id}.json"
    state = {
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        "source": "persona_chat",
        "schema_profile": schema_name,
        "desired_dataset_payload": desired,
        "field_origins": origins,
        "resource_plan": resource_plan if resource_plan is not None else [],
        "ckan": {"url": settings.ckan_url, "owner_org": desired.get("owner_org") or settings.ckan_owner_org},
        "warnings": [],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def route_after_persona(state: CkanRegistrationState) -> str:
    needs = state.get("persona_stop_reason") == STOP_NEEDS_CLARIFICATION
    under_cap = int(state.get("clarification_round") or 0) < CLARIFICATION_CAP
    if needs and under_cap and state.get("clarification_questions"):
        return "clarify"
    return "propose"


def build_persona_subgraph(
    settings: Settings,
    *,
    engine: EngineFn = run_persona_metadata_loop,
    checkpointer: Any = None,
):
    builder = StateGraph(CkanRegistrationState)
    builder.add_node("schema_select", make_schema_select_node(settings))
    builder.add_node("persona", make_persona_node(settings, engine=engine))
    builder.add_node("clarify", make_clarify_node())
    builder.add_node("propose", make_propose_node(settings))
    builder.add_edge(START, "schema_select")
    builder.add_edge("schema_select", "persona")
    builder.add_conditional_edges("persona", route_after_persona, {"clarify": "clarify", "propose": "propose"})
    builder.add_edge("clarify", "persona")
    builder.add_edge("propose", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
