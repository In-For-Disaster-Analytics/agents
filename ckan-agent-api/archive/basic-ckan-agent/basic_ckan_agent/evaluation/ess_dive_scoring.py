"""Availability-aware scoring for the ESS-DIVE round-trip eval.

Replaces the single "coverage %" with three honest numbers:

* derivable_score      - quality over only the fields the agent could know from
                         its inputs (files + DOI record), weighted + gated.
* catalog_completeness - how close the output is to the full gold record
                         (unavailable fields count as not-produced).
* fidelity_errors      - hard failures: mangled URLs, hallucinated facts,
                         invalid JSON, unsafe writes.

Each field is classified by availability and scored by its type (free text via
LLM judge; identifiers/dates/structured by normalized match; resource URLs by
canonical match with a hard fidelity check).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

from basic_ckan_agent.evaluation.extraction import _last_json_object

# Availability classes
PROVIDED = "provided"
INFERABLE = "inferable"
UNAVAILABLE = "unavailable"
EXTERNAL = "required_external"  # only counts when the DOI record is provided

# Field -> (type, base availability). "external" fields only become derivable
# when the DOI record is supplied AND the record carries the signal.
FIELD_DEFS: dict[str, tuple[str, str]] = {
    "name": ("slug", INFERABLE),
    "title": ("free_text", PROVIDED),
    "notes": ("free_text", INFERABLE),
    "type": ("exact", INFERABLE),
    "url": ("url_landing", EXTERNAL),
    "license_id": ("exact", EXTERNAL),
    "version": ("exact", EXTERNAL),
    "temporal_coverage_start": ("date", EXTERNAL),
    "temporal_coverage_end": ("date", EXTERNAL),
    "spatial": ("spatial", EXTERNAL),
    "tags": ("tags", INFERABLE),
    "extras": ("extras", EXTERNAL),
    "resources": ("resources", PROVIDED),
}

# Component weights for the derivable composite (renormalized over applicable ones).
COMPONENT_WEIGHTS = {"schema": 0.10, "free_text": 0.25, "structured": 0.25, "resources": 0.30, "tags": 0.10}
COMPONENT_FIELDS = {
    "free_text": ["title", "notes"],
    "structured": ["type", "url", "license_id", "version", "temporal_coverage_start", "temporal_coverage_end", "spatial", "extras", "name"],
    "tags": ["tags"],
    "resources": ["resources"],
}

REQUIRED_SCHEMA_FIELDS = ["name", "title", "notes", "resources"]


@dataclass
class FieldScore:
    field: str
    type: str
    availability: str
    score: float | None  # None = no gold value to compare
    status: str
    note: str = ""
    fidelity_errors: list[str] = field(default_factory=list)
    proposed: Any = None  # value the agent produced
    standard: Any = None  # gold value


@dataclass
class DatasetScore:
    name: str
    derivable_score: float = 0.0
    catalog_completeness: float = 0.0
    schema_valid: bool = True
    fidelity_errors: list[str] = field(default_factory=list)
    unavailable_fields: list[str] = field(default_factory=list)
    fields: dict[str, FieldScore] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    gates_applied: list[str] = field(default_factory=list)
    free_text_judgment: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #

def record_signals(record: dict | None) -> set[str]:
    """Which EXTERNAL fields the provided DOI record actually supports."""
    if not record:
        return set()
    ds = record.get("dataset", record) if isinstance(record, dict) else {}
    signals: set[str] = set()
    if ds.get("license"):
        signals.add("license_id")
    if ds.get("temporalCoverage") or ds.get("datePublished"):
        signals.update({"temporal_coverage_start", "temporal_coverage_end"})
    if ds.get("spatialCoverage"):
        signals.add("spatial")
    if record.get("viewUrl") or ds.get("@id"):
        signals.add("url")
    # DOI / repository / paper extras are always derivable from a record.
    signals.add("extras")
    # version is rarely explicit; leave unavailable unless present.
    return signals


def classify(field_name: str, record_provided: bool, signals: set[str]) -> str:
    _type, base = FIELD_DEFS[field_name]
    if base != EXTERNAL:
        return base
    if record_provided and field_name in signals:
        return PROVIDED
    return UNAVAILABLE


# --------------------------------------------------------------------------- #
# Free-text LLM judge
# --------------------------------------------------------------------------- #

_JUDGE_PROMPT = (
    "You are scoring generated CKAN dataset metadata for a public data catalog. Do not require exact wording.\n\n"
    "Source context (what the agent could see):\n{source}\n\n"
    "Reference title (gold): {gold_title}\nReference notes (gold): {gold_notes}\nReference tags (gold): {gold_tags}\n\n"
    "Generated title: {gen_title}\nGenerated notes: {gen_notes}\nGenerated tags: {gen_tags}\n\n"
    "Score each 1-5 (5 best):\n"
    "- title/notes on: faithfulness (no invented facts vs source), adequacy (captures the dataset's meaning), "
    "specificity (subject/geography/method/context), catalog usefulness, conciseness. A terse-but-accurate title "
    "is GOOD, not a failure.\n"
    "- tags on whether they are a reasonable, faithful, search-useful set for this dataset. Synonyms and different "
    "phrasings of the reference tags are fine (e.g. 'game camera' = 'wildlife-camera', 'Yakama River' covers "
    "'yakama-river-basin'); penalize invented/irrelevant tags or missing the main subjects.\n"
    "Set faithfulness_pass=false only if the title, notes, or tags introduce unsupported facts.\n"
    "Return raw JSON only, no code fences: "
    '{{"title_score": <1-5>, "notes_score": <1-5>, "tags_score": <1-5>, "faithfulness_pass": <true|false>, '
    '"comment": "<short>"}}'
)


def judge_free_text(generated: dict, gold: dict, source_context: str, model: Any) -> dict:
    prompt = _JUDGE_PROMPT.format(
        source=source_context[:6000] or "(none)",
        gold_title=gold.get("title", ""),
        gold_notes=gold.get("notes", ""),
        gold_tags=", ".join(sorted(_names(gold.get("tags")))) or "(none)",
        gen_title=generated.get("title", "") or "(none)",
        gen_notes=generated.get("notes", "") or "(none)",
        gen_tags=", ".join(sorted(_names(generated.get("tags")))) or "(none)",
    )
    try:
        # Accept either a LangChain model (.invoke -> message) or a DeepEval
        # wrapper (.generate -> str).
        if hasattr(model, "invoke"):
            raw = str(model.invoke(prompt).content)
        else:
            raw = str(model.generate(prompt))
        verdict = _last_json_object(raw) or {}
    except Exception as exc:  # judge failure should not crash the run
        return {"title_score": None, "notes_score": None, "faithfulness_pass": None, "comment": f"judge error: {exc}"}
    return verdict


# --------------------------------------------------------------------------- #
# Per-field-type scorers -> (score, status, note, fidelity_error)
# --------------------------------------------------------------------------- #

def _score_exact(gen: Any, gold: Any) -> tuple[float, str, str, str | None]:
    if _norm(gen) == _norm(gold):
        return 1.0, "match", "", None
    return 0.0, "differ", f"generated {gen!r} vs gold {gold!r}", None


def _score_date(gen: Any, gold: Any) -> tuple[float, str, str, str | None]:
    g, go = _date(gen), _date(gold)
    if g and g == go:
        return 1.0, "match", "", None
    if g and go and g[:7] == go[:7]:
        return 0.5, "partial", f"same month ({g} vs {go})", None
    return 0.0, "differ", f"{g or gen!r} vs {go or gold!r}", None


def _score_spatial(gen: Any, gold: Any) -> tuple[float, str, str, str | None]:
    gg, gogo = _geom(gen), _geom(gold)
    if not gg:
        return 0.0, "differ", "generated spatial is missing or invalid GeoJSON", None
    if not gogo:
        return None, "n/a", "no valid gold geometry", None
    if gg["type"] == gogo["type"] and _bbox_close(gg, gogo):
        return 1.0, "match", "geometry type + bbox align", None
    if gg["type"] == gogo["type"]:
        return 0.5, "partial", f"same type ({gg['type']}) but bbox differs", None
    return 0.25, "differ", f"type {gg['type']} vs {gogo['type']}", None


def _score_tags(gen: Any, gold: Any) -> tuple[float, str, str, str | None]:
    g, go = _names(gen), _names(gold)
    if not go:
        return None, "n/a", "no gold tags", None
    if not g:
        return 0.0, "missing", "agent produced no tags", None
    tp = len(g & go)
    precision = tp / len(g)
    recall = tp / len(go)
    f1 = 0.0 if (precision + recall) == 0 else round(2 * precision * recall / (precision + recall), 3)
    status = "match" if g == go else ("partial" if tp else "differ")
    note = f"F1={f1} (P={precision:.2f} R={recall:.2f}); missing={sorted(go - g)}; extra={sorted(g - go)}"
    return f1, status, note, None


def _score_extras(gen: Any, gold: Any) -> tuple[float, str, str, str | None]:
    g, go = _kv(gen), _kv(gold)
    if not go:
        return None, "n/a", "no gold extras", None
    if not g:
        return 0.0, "missing", "agent produced no extras", None
    matched = sum(1 for k, v in go.items() if k in g and _norm(g[k]) == _norm(v))
    present = sum(1 for k in go if k in g)
    score = round(matched / len(go), 3)
    status = "match" if matched == len(go) else ("partial" if present else "differ")
    return score, status, f"{matched}/{len(go)} extras matched (keys present: {present})", None


def _score_slug(gen: Any, gold: Any, gold_title: str) -> tuple[float, str, str, str | None]:
    if not gen:
        return 0.0, "missing", "agent produced no name", None
    if _norm(gen) == _norm(gold):
        return 1.0, "match", "", None
    looks_slug = bool(gen) and gen == gen.lower() and " " not in gen
    related = _jaccard(_tokens(gen), _tokens(gold_title)) >= 0.3 or _jaccard(_tokens(gen), _tokens(gold)) >= 0.3
    if related:
        return 0.5, "partial", "valid relation to title/source but not the gold slug" if looks_slug else "related to title but not slug-formatted", None
    return 0.25 if looks_slug else 0.0, "differ", "slug unrelated to gold/title", None


def score_resources(gen: Any, gold: Any, provided_urls: set[str]) -> tuple[float, str, str, list[str]]:
    gold_list = [r for r in (gold or []) if isinstance(r, dict)]
    gen_list = [r for r in (gen or []) if isinstance(r, dict)]
    if not gold_list:
        return None, "n/a", "no gold resources", []
    if not gen_list:
        return 0.0, "missing", "agent produced no resources", []

    provided_bases = {_basename(u) for u in provided_urls}
    remaining = list(gold_list)  # pair each generated resource once, avoiding basename collisions
    fidelity: list[str] = []
    url_scores: list[float] = []
    name_hits = fmt_hits = 0

    for gr in gen_list:
        gurl = gr.get("url", "")
        match = _best_gold_match(gurl, remaining)
        if match is not None:
            remaining.remove(match)
        if match and _canon(gurl) == _canon(match.get("url", "")):
            url_scores.append(1.0)
        elif match and _same_host(gurl, match.get("url", "")) and _basename(gurl) == _basename(match.get("url", "")):
            url_scores.append(0.5)
            # Path mutated despite the exact URL being provided in the manifest.
            if _basename(gurl) in provided_bases:
                fidelity.append(f"resource URL path mutated: {gurl} (expected {match.get('url')})")
        else:
            url_scores.append(0.0)
        if match:
            if _norm(gr.get("name")) == _norm(match.get("name")):
                name_hits += 1
            if _norm(gr.get("format")) == _norm(match.get("format")):
                fmt_hits += 1

    count_score = min(len(gen_list), len(gold_list)) / len(gold_list)
    url_score = round(sum(url_scores) / len(gold_list), 3)
    name_score = round(name_hits / len(gold_list), 3)
    fmt_score = round(fmt_hits / len(gold_list), 3)
    score = round(0.4 * url_score + 0.2 * count_score + 0.2 * name_score + 0.2 * fmt_score, 3)
    status = "match" if score >= 0.99 else ("partial" if score > 0 else "differ")
    note = f"count {len(gen_list)}/{len(gold_list)}, urls {url_score}, names {name_score}, formats {fmt_score}"
    return score, status, note, fidelity


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def score_dataset(
    generated: dict,
    gold: dict,
    *,
    record: dict | None,
    record_provided: bool,
    source_context: str,
    judge_model: Any,
    trajectory_actions: list[str] | None = None,
    agent_resources: list | None = None,
) -> DatasetScore:
    """Score one dataset.

    ``generated`` may have its ``resources`` pinned to the manifest by the caller;
    pass the agent's ORIGINAL resource list as ``agent_resources`` so the resource
    *fidelity* check still runs against what the agent actually produced.
    """
    result = DatasetScore(name=str(gold.get("name", "dataset")))
    signals = record_signals(record) if record_provided else set()
    provided_urls = {r.get("url") for r in (gold.get("resources") or []) if isinstance(r, dict) and r.get("url")}

    # Schema validity gate.
    result.schema_valid = bool(generated) and all(generated.get(f) not in (None, "", []) for f in REQUIRED_SCHEMA_FIELDS)

    # Free-text judge (one call).
    judgment = judge_free_text(generated, gold, source_context, judge_model)
    result.free_text_judgment = judgment

    for fname, (ftype, _base) in FIELD_DEFS.items():
        availability = classify(fname, record_provided, signals)
        gold_val = gold.get(fname)
        fs = _score_field(fname, ftype, generated.get(fname), gold_val, gold, judgment, provided_urls, availability)
        fs.proposed = generated.get(fname)
        fs.standard = gold_val
        result.fields[fname] = fs
        result.fidelity_errors.extend(fs.fidelity_errors)
        if availability == UNAVAILABLE:
            result.unavailable_fields.append(fname)

    # Resource fidelity against what the agent ACTUALLY produced (pre-pin):
    # catch mutated or omitted URLs even though the scored resources are pinned.
    if agent_resources is not None and gold.get("resources"):
        _, _, _, fids = score_resources(agent_resources, gold.get("resources"), provided_urls)
        result.fidelity_errors.extend(fids)
        missing = sum(1 for r in agent_resources if isinstance(r, dict) and not r.get("url"))
        if missing:
            result.fidelity_errors.append(f"agent omitted {missing} resource URL(s) (not copied from manifest)")

    # Components.
    result.components = _components(result, generated)

    # Derivable composite: components over available fields, weighted + renormalized.
    result.derivable_score = _weighted(result.components, applicable=set(result.components))
    # Completeness: include unavailable fields (re-score components counting them as 0).
    result.catalog_completeness = _completeness(result, generated, judgment)

    # Hard gates.
    result.derivable_score, result.gates_applied = _apply_gates(
        result.derivable_score, result, judgment, trajectory_actions or []
    )
    return result


def _score_field(fname, ftype, gen, gold_val, gold, judgment, provided_urls, availability) -> FieldScore:
    if ftype == "resources":
        score, status, note, fids = score_resources(gen, gold_val, provided_urls)
        return FieldScore(fname, ftype, availability, score, status, note, list(fids))
    if _empty(gold_val):
        return FieldScore(fname, ftype, availability, None, "n/a", "no gold value")
    # Agent explicitly declined a field it couldn't determine (e.g. "null"): not a hallucination.
    if _empty(gen):
        return FieldScore(fname, ftype, availability, 0.0 if availability != UNAVAILABLE else None,
                          "missing", "agent left field empty")

    if ftype == "free_text":
        key = "title_score" if fname == "title" else "notes_score"
        raw = judgment.get(key)
        score = None if raw is None else round(float(raw) / 5.0, 3)
        status = "n/a" if score is None else ("match" if score >= 0.8 else "partial" if score >= 0.5 else "differ")
        return FieldScore(fname, ftype, availability, score, status, f"judge {key}={raw}")
    if ftype == "slug":
        s, st, n, fe = _score_slug(gen, gold_val, str(gold.get("title", "")))
    elif ftype == "date":
        s, st, n, fe = _score_date(gen, gold_val)
    elif ftype == "spatial":
        s, st, n, fe = _score_spatial(gen, gold_val)
    elif ftype == "tags":
        raw = judgment.get("tags_score")
        if raw is not None:
            sc = round(float(raw) / 5.0, 3)
            _, _, f1note, _ = _score_tags(gen, gold_val)  # keep F1 as a reference
            s = sc
            st = "match" if sc >= 0.8 else "partial" if sc >= 0.5 else "differ"
            n = f"judge tags_score={raw}; {f1note}"
            fe = None
        else:
            s, st, n, fe = _score_tags(gen, gold_val)
    elif ftype == "extras":
        s, st, n, fe = _score_extras(gen, gold_val)
    else:  # exact, url_landing
        s, st, n, fe = _score_exact(gen, gold_val)

    fids = [fe] if fe else []
    # Hallucination check: produced a real value for a field unavailable from inputs.
    if availability == UNAVAILABLE and not _empty(gen) and st != "match":
        fids.append(f"possible hallucination: produced {fname} not available from inputs")
    return FieldScore(fname, ftype, availability, s, st, n, fids)


def _components(result: DatasetScore, generated: dict) -> dict[str, float]:
    comps: dict[str, float] = {"schema": 1.0 if result.schema_valid else 0.0}
    for comp, fields in COMPONENT_FIELDS.items():
        vals = [
            result.fields[f].score
            for f in fields
            if result.fields[f].availability != UNAVAILABLE and result.fields[f].score is not None
        ]
        if vals:
            comps[comp] = round(sum(vals) / len(vals), 3)
    return comps


def _completeness(result: DatasetScore, generated: dict, judgment: dict) -> float:
    # All gold fields with a value; unavailable still scored on the agent's output.
    scored = [fs.score for fs in result.fields.values() if fs.score is not None]
    base = sum(scored) / len(scored) if scored else 0.0
    # Mix in schema validity at the standard weight.
    return round(0.1 * (1.0 if result.schema_valid else 0.0) + 0.9 * base, 3)


def _weighted(components: dict[str, float], applicable: set[str]) -> float:
    weights = {c: COMPONENT_WEIGHTS[c] for c in applicable if c in COMPONENT_WEIGHTS}
    total = sum(weights.values())
    if not total:
        return 0.0
    return round(sum(components[c] * w for c, w in weights.items()) / total, 3)


def _apply_gates(score: float, result: DatasetScore, judgment: dict, actions: list[str]) -> tuple[float, list[str]]:
    from basic_ckan_agent.ckan.constants import WRITE_ACTIONS

    gates: list[str] = []
    if any(a in WRITE_ACTIONS for a in actions):
        gates.append("unsafe write action -> fail")
        return 0.0, gates
    if not result.schema_valid:
        score = min(score, 0.40)
        gates.append("invalid/incomplete JSON -> cap 0.40")
    if judgment.get("faithfulness_pass") is False:
        score = min(score, 0.70)
        gates.append("hallucinated core metadata -> cap 0.70")
    if any("resource url" in e.lower() for e in result.fidelity_errors):
        score = min(score, 0.75)
        gates.append("resource URL fidelity failure (mutated/omitted) -> cap 0.75")
    return round(score, 3), gates


# --------------------------------------------------------------------------- #
# normalization helpers
# --------------------------------------------------------------------------- #

_EMPTYISH = {"", "null", "none", "n/a", "na", "unknown", "not specified", "not available", "not provided", "nan"}


def _empty(v: Any) -> bool:
    if v is None or v == [] or v == {}:
        return True
    return isinstance(v, str) and v.strip().lower() in _EMPTYISH


def _norm(v: Any) -> str:
    return " ".join(str(v if v is not None else "").lower().split())


def _tokens(v: Any) -> set[str]:
    return {t for t in _norm(v).replace("-", " ").replace(",", " ").split() if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    return 0.0 if not a or not b else len(a & b) / len(a | b)


def _names(items: Any) -> set[str]:
    out: set[str] = set()
    for it in items or []:
        if isinstance(it, dict) and it.get("name"):
            out.add(_norm(it["name"]))
        elif isinstance(it, str):
            out.add(_norm(it))
    return out


def _kv(items: Any) -> dict[str, str]:
    return {_norm(it.get("key")): it.get("value") for it in (items or []) if isinstance(it, dict) and it.get("key")}


def _date(v: Any) -> str:
    s = str(v or "").strip()
    return s[:10] if len(s) >= 10 else s


def _geom(v: Any) -> dict | None:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return None
    if isinstance(v, dict) and v.get("type") and v.get("coordinates") is not None:
        return v
    return None


def _bbox_close(a: dict, b: dict, tol: float = 1.5) -> bool:
    ca, cb = _centroid(a), _centroid(b)
    if not ca or not cb:
        return False
    return abs(ca[0] - cb[0]) <= tol and abs(ca[1] - cb[1]) <= tol


def _centroid(geom: dict) -> tuple[float, float] | None:
    pts = _flatten_coords(geom.get("coordinates"))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _flatten_coords(coords: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if isinstance(coords, (list, tuple)):
        if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
            out.append((float(coords[0]), float(coords[1])))
        else:
            for c in coords:
                out.extend(_flatten_coords(c))
    return out


def _canon(url: str) -> str:
    return unquote(str(url or "")).rstrip("/").strip().lower()


def _same_host(a: str, b: str) -> bool:
    return urlparse(str(a)).netloc.lower() == urlparse(str(b)).netloc.lower()


def _basename(url: Any) -> str:
    return _path_segments(url)[-1] if _path_segments(url) else ""


def _path_segments(url: Any) -> list[str]:
    return [s.lower() for s in unquote(urlparse(str(url or "")).path).strip("/").split("/") if s]


def _best_gold_match(gen_url: str, golds: list[dict]) -> dict | None:
    """Pair a generated resource to the gold resource sharing the most trailing
    path segments (disambiguates files with the same basename in different dirs)."""
    gsegs = _path_segments(gen_url)
    if not gsegs:
        return None
    best, best_overlap = None, 0
    for r in golds:
        rsegs = _path_segments(r.get("url", ""))
        if not rsegs or rsegs[-1] != gsegs[-1]:  # require same filename to be a candidate
            continue
        overlap = 0
        for a, b in zip(reversed(gsegs), reversed(rsegs)):
            if a == b:
                overlap += 1
            else:
                break
        if overlap > best_overlap:
            best, best_overlap = r, overlap
    return best
