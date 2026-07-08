# Conversational Multi-Persona Metadata Authoring Agent

## Status

Implementing

_Increment 1 (DONE 2026-06-26): non-destructive foundation — `PersonaRegistry` +
`SchemaRegistry` (R6 hardening), seed persona `.md` files (`domain_expert`,
`data_curator`, `data_scientist`; evaluators emit the R1 `requires_human` field) +
`subside.yaml`/`generic_ckan.yaml`, settings (`personas_dir`, `schemas_dir`,
`runs_dir`, `default_schema_profile`, `CKAN_PERSONA_CHAT`). 15 new unit tests pass;
no regression to other collected tests. Purely additive — no existing behavior changed._

_Increment 2 (DONE 2026-06-26): persona engine moved into `app/personas/engine.py`
(R4) — pure/sync, registry- and schema-profile-driven, R1 `requires_human` structured
questions, R2 eager escalation (`stop_reason="needs_clarification"`), async-safe delay
(default 0), `runs_dir` audit trail, injectable `chat_fn`. Shared LLM helper added at
`app/llm.py`. 7 new engine tests pass (22 total new; 35 passing in the runnable suite).
No cross-repo change yet: the GAM batch path (`ckan-registration/orchestrate.py`) still
uses the original `persona_loop.py`, so its 117 tests are untouched. Pointing the GAM
batch at the new engine + snapshot regression tests is a later cross-repo increment
(Open Q3). `app/llm.py` ↔ `nodes._invoke_openai_chat` consolidation deferred to Increment 3._

_Increment 3a (DONE 2026-06-26): persona subgraph as a self-contained, compilable unit
in `app/agents/ckan_registration/persona_nodes.py` — `persona → (needs_clarification &
under cap) → clarify → persona; else → propose → END`. `clarify` reuses the real
LangGraph `interrupt()`/resume; on resume it applies the R3 split (org-level sticky vs.
dataset-specific). `propose` emits R6 field-origin-labeled review markdown and writes a
legacy-compatible `analyzed` state file for the existing dry-run/apply path. State fields
added. 4 subgraph tests pass (incl. a real interrupt→resume cycle); 26 feature tests, 39
runnable. The subgraph does NOT touch the legacy worker, so it compiles/tests cleanly._

_Increment 3b (DONE 2026-06-26): (1) `LegacyCkanWorker` is now lazy — the legacy module
loads on first `run()`, so `build_graph` succeeds without the cross-tree worker; this fixed
the two previously-uncollectable graph-import test modules. (2) The main graph routes
`analyze`/`revise` to the persona subgraph (`persona`/`clarify`/`propose`) when
`settings.persona_chat_enabled`, else the single-pass `metadata` node. (3) `nodes._invoke_openai_chat`
now delegates to `app/llm` (3 tests' patch targets updated to `app.llm`, faithfully).
Result: 62 passing. The only remaining 3 failures are `test_ckan_guardrails`, which has its
own hardcoded loader for `<parents[2]>/ckan-registration/ckan_agent.py` (absent in this
`/Volumes` checkout); they test the untouched legacy worker and fail purely on repo layout._

_Increment 4 (DONE 2026-06-26): extractor migration. Copied `basic-ckan-agent/.../files/`
extractors into `app/files/extractors/` (archive, image, json_data, pdf, spatial, tabular,
text — spatial import rewired to `app.files`), ported `safety.py` (decoupled from
`basic_ckan_agent.settings`; reads `CKAN_AGENT_MAX_FILE_BYTES`) and `results.py`, and added
`app/files/analyze.py` — `analyze_path` (extension-routed, safety-checked per S-4),
`analyze_request_files` (resolves files/upload_dirs), `build_file_inventory`. The `persona`
node now builds real author evidence (`_gather_evidence`) from extractor reports + a file
inventory instead of the message/URL stub. The tool-calling layer (`tools.py`, `catalog.py`,
`tool_catalog.yaml`, `schemas.py`) was NOT migrated — it belongs to a different agent pattern
and is archived in Increment 5. 6 new extractor/dispatcher tests; 68 passing (only the
unrelated `test_ckan_guardrails` layout artifact remains)._

_Increment 5 (DONE 2026-06-26): reversible archival. Moved `basic-ckan-agent/` and its two
tests (`tests/test_basic_ckan_file_tools.py`, `tests/test_basic_ckan_smoke.py`) → `archive/`
(all were untracked, so plain `mv`, fully reversible); added `archive/README.md`. `app/`
imports nothing from `basic_ckan_agent`, and `testpaths=["tests"]` means the archived tests
are not collected. Suite: 60 passing (down from 68 only because the 8 archived `basic_ckan`
tests left the live suite; no new failures; same 3 unrelated `ckan_guardrails` artifacts).
NOT archived (still live default while `CKAN_PERSONA_CHAT` is off): the single-pass metadata
path. NOT done (needs separate sign-off): hard `git rm` of caches/`.DS_Store` (housekeeping)._

_Remaining work: (a) archive the single-pass metadata path once persona-chat becomes the
default; (b) cross-repo — point `ckan-registration/orchestrate.py` at `app/personas/engine`
and add GAM snapshot regression tests so the 117 GAM tests stay green on the shared engine;
(c) resolve the `test_ckan_guardrails` repo-layout artifact (co-locate `ckan-registration` or
fix that test's hardcoded path)._

> **Environment finding (blocks Increment 2/3 test runs):** in this working tree the
> `ckan-registration` worker is at `/Users/wmobley/Documents/GitHub/agents/ckan-registration`,
> but `Settings.legacy_ckan_registration_dir` resolves to `REPO_ROOT/ckan-registration`
> (`/Volumes/Macintosh HD - Data/Github/agents/ckan-registration`), which does not exist
> here. `graph.py` builds the graph at import time, so `test_metadata_report_graph.py`,
> `test_openai_compat.py`, and `test_ckan_guardrails.py` fail to collect/run independent of
> this work. Pre-existing; must be resolved (set `CKAN_AGENT_LEGACY_DIR`, co-locate the repos,
> or — better, per R4 — move the engine in so the graph no longer needs the cross-tree worker)
> before the graph-rewiring increment can be validated end to end._

## Objective

Turn the `ckan-registration` notebook workflow into an interactive chat agent in
which a **Domain Expert** persona reads supplied files / software / PDFs / data,
designs CKAN metadata, **asks the human user for clarification when needed**,
otherwise **discusses with a Data Curator and a Data Scientist** persona, and then
presents proposed metadata for human approval before submitting to CKAN on the
user's behalf.

## User need

A domain expert wants to register a dataset by chatting, not by editing a notebook
or composing JSON command payloads. They want to:

1. Drop in files / a directory / a PDF / a source URL.
2. Let an expert agent draft the metadata and self-critique it (curator + scientist).
3. Be asked focused questions **only** when the agent genuinely cannot determine a
   field from the sources.
4. Review the proposed metadata (and a CKAN dry-run diff) before anything is written.
5. Approve once, and have the agent submit to CKAN.

This must work for **general datasets**, not only TWDB GAMs (per scoping decision).

## Current code/system summary

Almost every building block already exists; they are just not wired together into
one conversational, human-in-the-loop flow. Three subsystems are relevant:

### 1. `ckan-registration/` — the worker backend
- **Three personas already implemented** in
  [persona_loop.py](../../../ckan-registration/src/gam_registration/persona_loop.py):
  `DOMAIN_EXPERT_PROMPT` (author), `DATA_CURATOR_PROMPT` (FAIR reviewer),
  `DATA_SCIENTIST_PROMPT` (usability reviewer). `run_persona_metadata_loop` runs
  author → both evaluators (in parallel) → convergence check, up to `max_rounds`.
- **Gap A (human loop):** evaluator `questions` are routed only **back to the LLM
  author** for another round
  ([persona_loop.py L752](../../../ckan-registration/src/gam_registration/persona_loop.py#L752)).
  Unresolved questions accumulate as `outstanding_questions`
  ([orchestrate.py L733](../../../ckan-registration/src/gam_registration/orchestrate.py#L733))
  and the run simply does not converge. **There is no pause-and-ask-the-human step.**
- **Gap B (GAM-specific):** the persona prompts are hardwired to TWDB GAMs and the
  `subside_dataset` schema (mandatory `collection_method = "Model Output"`,
  `categories` must include `"Groundwater"`, etc.).
- The persona loop is only ever called from the **GAM batch orchestrator**
  (`orchestrate.run_registration`), never from the interactive path.
- An approval-gated CKAN write path exists end to end
  (`ckan_agent.py` analyze → dry-run → revise → apply, gated on `approval == "REGISTER"`).

### 2. `ckan-agent-api/` — the chat surface (LangGraph + FastAPI)
- LangGraph state machine in
  [graph.py](../app/agents/ckan_registration/graph.py): `intake → {metadata,
  dry-run, approval→apply, show}`. OpenAI-compatible `/v1/chat/completions` endpoint.
- **Human-in-the-loop already works here**: `make_approval_node`
  ([nodes.py L2048](../app/agents/ckan_registration/nodes.py#L2048)) calls LangGraph
  `interrupt(...)`; the runner surfaces it as `requires_action` and `resume()` feeds
  the user's reply back via `Command(resume=...)`
  ([graph.py L106](../app/agents/ckan_registration/graph.py#L106)). **This is exactly
  the mechanism the clarification step needs — it just is not used for metadata yet.**
- **Gap C (shallow metadata):** the `metadata` node (`make_file_metadata_node` →
  `build_file_metadata_report`) uses a **single LLM pass** over `metadata_guide.md`
  ([nodes.py L1916](../app/agents/ckan_registration/nodes.py#L1916)). It does **not**
  call the persona loop — so the chat path today has no curator/scientist discussion.
- It already has solid general-purpose **file readers** (csv/tsv, json/geojson,
  ipynb, zip, pdf, html, text) in `nodes.py`.

### 3. `ckan-agent-api/basic-ckan-agent/` — richer extractors (optional reuse)
- A separate agent with a `files/extractors/` catalog (pdf, tabular, spatial, image,
  archive, json) and an evaluation harness. Candidate source for stronger
  general-dataset file understanding.

**Bottom line:** ~80–90% exists. The new work is three seams: (A) a human-clarify
interrupt, (B) generalized (schema-parameterized) personas, (C) wiring the persona
loop into the chat `metadata` node.

## Proposed design

Add a persona-discussion path to the LangGraph agent that reuses the existing
interrupt/resume mechanism for human clarification.

### New / changed graph shape

```
intake
  └─(analyze/revise)→ author        (Domain Expert drafts metadata)
                        │
                        ▼
                     discuss         (Curator + Scientist evaluate, in parallel)
                        │
              ┌─────────┴──────────┐
       converged / no            blocking question
       answerable Qs          NOT resolvable from sources
              │                      │
              ▼                      ▼
           propose            clarify (interrupt → ask user)
        (review markdown)            │  resume with answers
              │                      └────────► back to author
              ▼
           dry-run → approval → apply   (unchanged, already gated)
```

- **author node** — wraps `run_persona_metadata_loop`'s author call (or the whole
  loop with `max_rounds` internal rounds). Produces candidate metadata + `_gap_`
  annotations.
- **discuss node** — runs the FAIR + usability evaluators. Splits their `questions`
  into two buckets:
  - *source-derivable* (the author missed something present in the files) → loop
    internally (existing behavior).
  - *not derivable from any source and lacking a `_gap_` resolution* → escalate to
    **clarify**.
- **clarify node** — `interrupt(...)` with the specific question(s), exactly like
  `make_approval_node`. **Escalation is eager: the moment the discuss node produces
  the first blocking question that is not derivable from any source (and has no
  `_gap_` resolution), the loop stops and asks the user** — it does *not* burn the
  full `max_rounds` first (decision 2026-06-26). The user's `resume` answers are
  injected as authoritative `organizational_metadata` / dataset overrides and fed
  back to the author. A clarification-round cap prevents infinite loops.
- **propose node** — emits the review markdown (proposed package body + outstanding
  notes) and ends the turn; the user then asks for `dry run` → `register`.

### Clarification answers persist per thread

Answers the user gives during `clarify` are written into the session's saved state
as `organizational_metadata` (decision 2026-06-26). Because the LangGraph thread is
checkpointed (sqlite), those answers are **reused as authoritative defaults for
subsequent datasets registered in the same thread** — e.g. once the user states the
license or maintainer, later datasets in that conversation inherit it and the
personas do not re-ask. They remain overridable per dataset.

### User-extensible personas (markdown + frontmatter, Claude-style scaffolding)

Personas become **user-authored markdown files with YAML frontmatter**, discovered
from a directory — the same scaffolding pattern Claude Code uses for agents/skills
(decision 2026-06-26). A non-developer can drop in a new persona without touching
Python.

```
app/personas/
  domain_expert.md
  data_curator.md
  data_scientist.md
  <user-added>.md
```

Each file:

```markdown
---
name: data-curator
description: FAIR-principles reviewer; checks findable/accessible/interoperable/reusable.
role: evaluator            # author | evaluator
when_to_use: Always run as a reviewer after the author drafts metadata.
enabled: true
---
<system prompt body — may use {{schema_fields}}, {{controlled_vocab}}, {{defaults}} tokens>
```

- A new **`PersonaRegistry`** (extends today's `{{token}}` `PromptRegistry`) parses
  frontmatter, validates required keys, filters `enabled`, and groups by `role`. The
  three current persona prompts in `persona_loop.py` are extracted verbatim into the
  three seed markdown files (GAM-specifics moved to schema tokens, not the prompt body).
- The graph builds the loop from whatever personas the registry yields: one `author`
  + N `evaluator`s. Adding a 4th reviewer = adding a markdown file.

### Configurable schema / controlled-vocab directory (default: SUBSIDE)

The schema and controlled vocabularies are **not** hardcoded. They live in a
configurable directory of YAML "schema profiles" (decision 2026-06-26):

```
app/schemas/
  subside.yaml          # default profile
  generic_ckan.yaml
  <user-added>.yaml
```

Each profile carries a **`when_to_use` description** so the agent (and the user) can
pick the right one, plus the field list, controlled vocab, and hard defaults:

```yaml
name: subside
description: TWDB/SUBSIDE groundwater datasets (GAMs, subsidence models).
when_to_use: >
  Use for Texas groundwater availability models, subsidence datasets, or anything
  destined for the subside_dataset CKAN type.
dataset_type: subside_dataset
defaults:                 # applied unconditionally (the old GAM defaults)
  collection_method: Model Output
  categories: [Groundwater]
controlled_vocab:
  categories: [Boundaries, Groundwater, Natural Hazards, Planning, Water Quality, Water Use]
  collection_method: [Administrative Record, Instrumentation Measurement, ...]
fields:                   # key, label, required, guidance — fed to the author/evaluators
  - {key: temporal_coverage_start, guidance: "ISO-8601 year or date"}
  ...
```

- A **`SchemaRegistry`** discovers profiles, exposes `when_to_use` for selection, and
  renders the `{{schema_fields}}`/`{{controlled_vocab}}`/`{{defaults}}` tokens into
  the persona prompts.
- **Default = `subside`** for now. The seed `subside.yaml` is derived from the existing
  `ckan-registration/schema/subside_dataset.proposed.yaml` (fields + vocab + the GAM
  defaults currently hardcoded in `orchestrate._GAM_DEFAULTS` and the author prompt).
- Selection: explicit (`request.schema = "subside"`), else a default setting; later we
  can let the agent suggest a profile from `when_to_use`. Out of scope for v1: auto-detect.

### File extractors migrated in now (incl. spatial + image)

Per decision 2026-06-26, `basic-ckan-agent`'s extractor package is migrated into the
main app **now** (not deferred), replacing the inline `_analyze_file` readers in
`nodes.py`:

- Move `basic-ckan-agent/basic_ckan_agent/files/` → `app/files/` (extractors:
  `archive, image, json_data, pdf, spatial, tabular, text`, plus `catalog.py`,
  `results.py`, `safety.py`, `schemas.py`, `tools.py`, `tool_catalog.yaml`).
- The `author` node uses these extractor reports as its evidence. Spatial (`rasterio`
  CRS/bounds) and image (header dims) become available for general datasets.
- `rasterio` is an **optional** dependency (extractor already degrades gracefully with
  a `dependency_missing` note), so the base install is unaffected.

### Reuse, don't rebuild

- CKAN dry-run, approval gate, and apply: **unchanged**.
- `persona_loop.run_persona_metadata_loop` stays the engine; it gains (a) prompt/
  schema parameterization, and (b) an **eager escalation return path** that stops and
  surfaces the first non-derivable blocking question instead of silently not converging.

## Consolidation & archival plan

`ckan-agent-api` is its own git repo and becomes **the single agent** (decision
2026-06-26). `basic-ckan-agent` and the superseded single-pass metadata path are
folded in or archived. **All moves use `git mv` into an `archive/` tree — nothing is
hard-deleted in this spec.** The concrete deprecation/deletion list (below) requires
**explicit user sign-off** before execution (destructive-change gate).

**Migrate (keep, move into the app):**
- `basic-ckan-agent/basic_ckan_agent/files/` → `app/files/` (extractors + catalog +
  safety + schemas). This is the new file-understanding layer.

**Archive (move to `archive/`, not deleted):**
- The single-pass metadata path once the persona path is default:
  `app/prompts/ckan_registration/metadata_guide.md` and the
  `build_file_metadata_report` / `_prompt_guided_metadata` / `_guess_dataset_metadata`
  helpers in `nodes.py` (kept reachable behind the rollback flag until the new path
  ships green, then archived).
- The rest of `basic-ckan-agent/` not migrated (its standalone CKAN client, tools,
  CLI) **and its `evaluation/` harness** — all archived (decision 2026-06-26).

**Candidate deletions (require explicit approval — not done in this spec):**
- `__pycache__`, `.pytest_cache`, `.ruff_cache`, stray `.DS_Store`.
- `basic-ckan-agent/` duplicated extractor copies after migration.

The exact file-by-file deprecation list will be produced as a reviewable checklist
before any `git rm`, per the approval gate.

## Files likely affected

| File / dir | Change |
|---|---|
| `ckan-registration/src/gam_registration/persona_loop.py` | Parameterize persona prompts via schema tokens (remove hardcoded GAM defaults); add eager `needs_clarification` return path (first non-derivable blocking question) distinct from `converged` / `max_rounds`. GAM behavior preserved via the `subside` schema profile + params. |
| `ckan-agent-api/app/personas/` (new) | `domain_expert.md`, `data_curator.md`, `data_scientist.md` — markdown + YAML frontmatter; user-extensible. |
| `ckan-agent-api/app/schemas/` (new) | `subside.yaml` (default), `generic_ckan.yaml` — schema profiles with `when_to_use`, fields, controlled vocab, defaults. |
| `ckan-agent-api/app/prompts/__init__.py` | Extend into `PersonaRegistry` (frontmatter parse, role grouping, `enabled` filter) + `SchemaRegistry`. |
| `ckan-agent-api/app/files/` (new, migrated) | Extractor package moved from `basic-ckan-agent`; spatial + image included. |
| `ckan-agent-api/app/agents/ckan_registration/graph.py` | Add `author`, `discuss`, `clarify`, `propose` nodes + edges; route `analyze/revise` through them (behind rollout flag). |
| `ckan-agent-api/app/agents/ckan_registration/nodes.py` | New node factories; `clarify` reuses `make_approval_node`'s `interrupt()` pattern; author uses `app/files` extractors; archive single-pass helpers after cutover. |
| `ckan-agent-api/app/agents/ckan_registration/state.py` | Add `candidate_metadata`, `evaluator_verdicts`, `clarification_questions`, `clarification_round`, `organizational_metadata`, `schema_profile`. |
| `ckan-agent-api/app/agents/ckan_registration/schemas.py` | Extend request/resume schemas: `schema` profile selector, `clarifications` resume payload. |
| `ckan-agent-api/app/settings.py` | `personas_dir`, `schemas_dir`, default schema profile, `CKAN_PERSONA_CHAT` flag, optional `rasterio`. |
| `ckan-agent-api/environment.yml` / deps | Add optional `rasterio` (graceful-degrade if absent). |
| `archive/` (new) | Superseded single-pass path + unmigrated `basic-ckan-agent` parts. |
| Tests | New unit + graph tests (see Test plan); migrate relevant extractor tests. |

## API/schema changes

- **No new HTTP endpoints.** Clarification reuses the existing
  `requires_action` (interrupt) + `POST /runs/{thread_id}/resume` contract.
- New `requires_action.type`: `"metadata_clarification_required"`, carrying the
  question list and the field(s) each maps to. Clients resume with
  `{"clarifications": {<field>: <answer>, ...}}` or free-text `message`.
- `CkanRegistrationState` gains the fields listed above (additive, `total=False`).

## Data flow

1. User sends files/dir/URL + message → `intake` normalizes, routes `analyze`.
2. `author` drafts candidate metadata from extractor reports + (optional) schema.
3. `discuss` runs curator + scientist; buckets their questions.
4. If any blocking, non-source-derivable questions → `clarify` interrupts; user
   answers via `resume`; answers become authoritative `organizational_metadata`;
   back to `author` (capped rounds).
5. On convergence (or cap) → `propose` returns review markdown.
6. User: `dry run` → existing diff; `REGISTER` → existing gated apply to CKAN.

## Risks and tradeoffs

- **Latency / cost:** persona loop is 3 LLM calls/round vs. 1 today; clarification
  adds round-trips. Mitigate with the existing `max_rounds` cap and a clarification
  cap, and by allowing the single-pass path to remain for trivial datasets.
- **Over-asking the user:** the whole point of the `_gap_` / re-raise-prevention
  logic is to avoid noise. The bucketing heuristic (source-derivable vs. not) is the
  riskiest new logic and must be tested against the re-raise constraints already in
  the persona prompts.
- **Regression risk to GAM batch path:** parameterizing the prompts must not change
  GAM output. Mitigate by making GAM defaults explicit params and snapshot-testing
  existing GAM fixtures.
- **Two file-reader implementations** (`ckan-agent-api` nodes vs. `basic-ckan-agent`):
  v1 reuses the in-graph readers; consolidating is deferred.
- **Cross-repo coupling:** `ckan-agent-api` already imports the `ckan-registration`
  worker; adding a persona-loop import deepens that dependency. Acceptable given the
  existing `legacy_worker` bridge.

## Alternatives considered

1. **Single richer LLM pass (no personas in chat).** Cheaper, but loses the
   curator/scientist discussion the user explicitly asked for. Rejected.
2. **Auto-fill gaps without asking the user.** Faster, but fabricates metadata —
   violates the existing "do not invent" rules and the user's clarification ask. Rejected.
3. **Build a brand-new agent from scratch.** Throws away the working interrupt,
   approval gate, extractors, and personas. Rejected.
4. **Notebook widget / standalone SDK agent instead of the LangGraph service.**
   The user pointed to `./agents` (the existing `ckan-agent-api`); building there
   reuses the deployed chat surface. Chosen.

## Test plan

- **persona_loop unit tests:** prompt parameterization preserves GAM behavior
  (snapshot existing fixtures); new escalation return path fires only for
  non-source-derivable blocking questions; `_gap_` re-raise prevention still holds.
- **graph tests:** `analyze → discuss → clarify` raises an interrupt;
  `resume` with answers reaches `author` and then `propose`; clarification cap
  terminates; `propose → dry-run → approval → apply` still gated on `REGISTER`.
- **General-dataset fixtures:** add non-GAM samples (a CSV, a notebook, a PDF) and
  assert sensible proposed metadata + at most N clarification questions.
- Run existing suites: `ckan-registration` (117 tests) and `ckan-agent-api` pytest —
  must stay green.

## Documentation plan

- Update `ckan-agent-api/README.md`: new clarification `requires_action` type and
  resume payload; note that `analyze` now runs the persona discussion.
- Document the templated persona prompts in `app/prompts/ckan_registration/`.
- Add a short "conversational registration" walkthrough.

## Rollout/rollback plan

- Gate the new path behind a setting (e.g. `CKAN_PERSONA_CHAT=1`); default off until
  green. The single-pass `metadata` node remains the fallback.
- Roll out: enable in staging → run general-dataset + GAM fixtures → enable default.
- Rollback: flip the flag off; graph reverts to the existing single-pass `metadata`
  node with no schema/state migration needed (new state fields are optional).

## Open questions

_Resolved (2026-06-26) — see Decisions:_
- ~~Escalate eagerly vs. after max_rounds?~~ → **Eager**, on first non-derivable blocking question.
- ~~Default schema/vocab?~~ → **SUBSIDE**, via a configurable `app/schemas/` directory of YAML profiles.
- ~~Persist clarifications per thread?~~ → **Yes**, as reusable `organizational_metadata`.
- ~~Pull in spatial/image extractors now?~~ → **Yes**, migrate the whole extractor package now.

_Resolved by team discourse + user decisions (2026-06-26):_
- ~~`basic-ckan-agent/evaluation/` harness?~~ → **Archive it.**
- ~~Bucketing classifier?~~ → **R1: `requires_human` evaluator output field.**
- ~~org_metadata bleed default?~~ → **R3: org-level only; re-ask dataset-specific.**
- ~~Persona engine home / cross-repo coupling?~~ → **R4: move into ckan-agent-api.**
- ~~Schema source of truth?~~ → **R5: `app/schemas/subside.yaml` canonical.**
- ~~Clarification reset?~~ → **R3: yes, user-clearable mid-thread.**

_Still open (non-blocking; can be settled during implementation):_
1. **Schema profile auto-suggest** — v1 uses explicit `request.schema` + a default setting;
   agent *suggesting* a profile from `when_to_use` is deferred. Confirm deferral is OK.
2. **Persona `role` vocabulary** — limited to `author` / `evaluator` for v1, or open-ended? (Proposed: limited.)
3. **GAM batch migration timing** — does `orchestrate.py` import the engine back immediately (R4),
   or does the whole GAM batch path move into the agent in a follow-up?

## Design revisions after team discourse (2026-06-26)

Team discourse (architect, skeptic, security-reviewer, tester) returned a unanimous
**REVISE**. The direction was endorsed; the following concrete commitments close the
blocking gaps. They supersede any conflicting earlier text above.

### R1 — `requires_human` is an evaluator output field, not a post-hoc classifier
The "source-derivable vs. non-derivable" split is **not** inferred in the `discuss`
node. Instead the evaluator JSON schema gains a structured flag per question:

```json
{ "verdict": "revise",
  "questions": [
    {"field": "license_id", "question": "...", "requires_human": true,
     "reason_not_derivable": "no license stated in any source"} ],
  "recommendations": [...] }
```

- `DATA_CURATOR_PROMPT` / `DATA_SCIENTIST_PROMPT` are updated to emit `requires_human`
  per question: `true` only when the field is absent from **all** sources (files,
  PDF, landing page, file_inventory, organizational_metadata) AND lacks a `_gap_`
  resolution. The existing re-raise-prevention rules are tightened to set
  `requires_human: false` for `_gap_`-annotated fields.
- `EvaluatorVerdict.questions` becomes a list of structured objects (back-compat:
  a string is coerced to `{question: <str>, requires_human: false}`).
- `discuss` escalates to `clarify` **iff** any question has `requires_human == true`.
  Source-derivable questions stay in the internal loop (existing behavior).
- This guards both failure modes the skeptic raised: it won't nag the user with
  loop-resolvable questions, and it won't surface genuinely unanswerable `_gap_`
  questions to the user.

### R2 — Eager escalation, but only on a true `requires_human` signal
Escalation remains eager (user decision) — fire on the **first** question with
`requires_human == true`, without exhausting `max_rounds`. Because R1 makes that flag
reliable, eager escalation no longer risks nagging. Source-derivable revises still get
internal author rounds first.

### R3 — `organizational_metadata` scope (resolves cross-dataset bleed)
**Decision (user 2026-06-26): persist org-level fields only; re-ask dataset-specific.**
- Persisted & reused across datasets in a thread: `owner_org`, `maintainer`,
  `maintainer_email`, `data_contact_email`.
- **Always re-derived/re-asked per dataset:** `license_id`, `author`, `author_email`,
  `temporal_coverage_*`, `spatial`, `notes`, `title`, etc.
- State keys: `org_metadata` (sticky, thread-scoped) vs. `dataset_clarifications`
  (per-dataset, cleared when a new dataset registration starts in the thread). Merge
  semantics: org-level merges (last-writer-wins per key); dataset-level resets.
- A user can clear sticky values mid-thread (e.g. "forget the maintainer") — handled
  in `intake` by a reset directive (Open Q4 resolved: yes, clearable).

### R4 — Persona engine moves INTO ckan-agent-api (resolves cross-repo coupling)
**Decision (user 2026-06-26).** `run_persona_metadata_loop` + dataclasses move to
`ckan-agent-api/app/personas/engine.py` (prompts externalized to `app/personas/*.md`).
- The GAM batch path (`ckan-registration/orchestrate.py`) imports the engine back from
  the agent package (or is migrated later); no cross-filesystem `sys.path` injection.
- **Async safety:** the engine's `LLM_CALL_DELAY_SECONDS` synchronous `time.sleep`
  must not block the FastAPI event loop. Persona/LLM nodes run via
  `asyncio.to_thread` (or the delay defaults to `0` in the service and is non-zero
  only in batch). Required, per architect/skeptic.
- **Audit trail:** add `runs_dir` to `Settings`; the engine writes transcripts there
  (today it writes to cwd and silently fails in a service).

### R5 — Schema single source of truth
**Decision (user 2026-06-26): `app/schemas/subside.yaml` is canonical.** The GAM batch
path reads SUBSIDE fields/defaults from it; `ckan-registration/schema/subside_dataset.proposed.yaml`
(the ckanext-scheming deploy artifact) is regenerated from / validated against it, not
hand-maintained in parallel. A consistency test asserts they don't drift.

### R6 — Security commitments (from security-reviewer; blocking ones folded in)
- **Persona loader hardening:** `PersonaRegistry` resolves each file and rejects any
  path escaping `personas_dir` (symlink/`..` guard); validates frontmatter keys; fails
  **loudly at startup** on parse error / missing `role` / >1 `author`. Persona bodies
  are flagged for review if they contain the approval token literal.
- **`yaml.safe_load` only** in `SchemaRegistry` (never `yaml.load`). Stated requirement + test.
- **Dry-run field-origin labeling:** the dry-run/proposal output labels each field
  `user-supplied | llm-derived | schema-default` so "REGISTER" is informed consent.
  (Amends the earlier "dry-run unchanged" note.)
- **`state_path` confinement:** fix the pre-existing `_state_path_from_request` to
  reject paths outside `state_dir` before the new flow writes PII into state.
- **Checkpoint store:** move the sqlite checkpoint off `/tmp` to an app-owned dir with
  `0600`; document it. Provenance: log schema-profile name + persona-file hashes per
  registration so a bad file's blast radius is recoverable.
- **Extractors:** carry `basic-ckan-agent/files/safety.py` along; enforce an
  uncompressed-size limit before `fiona`/zip extraction.

### R7 — Graph & registry details
- Explicit conditional edges: `discuss → clarify` (any `requires_human`) else
  `discuss → propose`; `clarify → author` (resume, capped) else `clarify → propose`.
- **Clarification cap:** max 3 interrupts per dataset OR 2 per field, whichever first;
  on cap, `propose` with outstanding notes (no silent incomplete write).
- Extractor output-interface contract documented + asserted so swapping the inline
  `_analyze_file` readers for `app/files/` doesn't silently change author evidence.

## Decisions

- **Scope = general datasets** (user, 2026-06-26). Personas must be schema-parameterized,
  not GAM-hardcoded.
- **Surface = existing `ckan-agent-api` LangGraph service** (user pointed to `./agents`,
  2026-06-26). Reuse its interrupt/resume + approval gate rather than build new.
- **Eager clarification escalation** (user, 2026-06-26): stop and ask the user the moment
  the first non-source-derivable blocking question appears, rather than exhausting `max_rounds`.
- **Default schema = SUBSIDE, via a configurable directory** (user, 2026-06-26): YAML schema
  profiles under `app/schemas/`, each with a `when_to_use` description so more can be added;
  `subside` is the default profile, seeded from `subside_dataset.proposed.yaml`.
- **Persist clarification answers per thread** (user, 2026-06-26): store as
  `organizational_metadata` in checkpointed session state; reuse for later datasets in the thread.
- **Migrate `basic-ckan-agent` extractors now** (user, 2026-06-26), including spatial + image.
- **Consolidate into one agent** (user, 2026-06-26): `ckan-agent-api` is THE agent; migrate what's
  needed, archive deprecated files (old single-pass metadata path, unmigrated `basic-ckan-agent`),
  via `git mv` to `archive/`. Hard deletions need a separate explicit approval.
- **User-extensible personas via markdown + frontmatter** (user, 2026-06-26): mirror Claude's
  agent/skill scaffolding so users can add persona `.md` files (and schema YAML) without code.
- **Persona engine moves into `ckan-agent-api`** (user, 2026-06-26, post-discourse R4): no
  cross-repo import; GAM batch path imports the engine back.
- **`organizational_metadata` = org-level sticky, dataset-specific re-asked** (user, 2026-06-26,
  R3): prevents license/author bleed across datasets in a thread; sticky values are user-clearable.
- **`app/schemas/subside.yaml` is the canonical SUBSIDE schema** (user, 2026-06-26, R5); the
  ckanext-scheming deploy YAML is derived/validated against it.
- **`requires_human` evaluator flag drives escalation** (post-discourse R1); `yaml.safe_load`,
  persona path-confinement + loud startup validation, and dry-run field-origin labeling are
  required security commitments (R6).

## User feedback / decisions

- 2026-06-26: User asked how much of the notebook can become a chat agent where a
  domain expert designs metadata, asks the user for clarification, discusses with a
  curator and data scientist, and proposes metadata before submitting to CKAN.
- 2026-06-26: User selected interface = `./agents` (the `ckan-agent-api` service) and
  scope = general datasets.
- 2026-06-26: User resolved all four open questions and expanded scope — eager escalation;
  SUBSIDE default via a configurable YAML schema directory (with `when_to_use`); persist
  clarifications per thread; migrate spatial/image extractors now; consolidate everything into
  `ckan-agent-api` (archive deprecated files, incl. old personas/single-pass path); add a
  user-extensible markdown+frontmatter persona directory using Claude-style agent/skill scaffolding.
- 2026-06-26: Team discourse (architect, skeptic, security-reviewer, tester) returned unanimous
  REVISE. Findings folded in as R1–R7 (see "Design revisions after team discourse").
- 2026-06-26: User decisions on the three genuine forks — engine **moves into ckan-agent-api**;
  org_metadata **org-level only, re-ask dataset-specific**; **`app/schemas/subside.yaml` canonical**.
- _Spec revised post-discourse. Awaiting final approval (and sign-off on the archival/deletion
  list) before implementation._
