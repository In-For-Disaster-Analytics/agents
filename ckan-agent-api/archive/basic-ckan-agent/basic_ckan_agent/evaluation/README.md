# CKAN Agent — Local Evaluation Suite (DeepEval)

Compare prompt/model configurations of the CKAN metadata LangGraph agent on a
regression suite — **fully local, no external eval service**. The agent *and*
every LLM judge run on your own endpoint (`OPENAI_BASE_URL` / `CKAN_LLM_MODEL`,
e.g. the TACC Llama deployment).

The goal is **not** to force one exact title or description, but to measure
whether generated metadata is **faithful, searchable, specific, useful, and
safe** for a public CKAN catalog — and whether the agent used the right tools
without performing unsafe writes.

## What gets measured

| Layer | Metric | Purpose |
|-------|--------|---------|
| Deterministic | `title_basic_quality`, `description_basic_quality`, `must_mention_terms` | Cheap, reproducible gates for obvious failures (empty, placeholder, filename-only, too short/long, filler, missing required terms). |
| LLM-as-judge (GEval) | `Title Quality`, `Description Quality` | Rubric scores normalized to 0–1 (mapped back to 1–5 for the gate). No exact match required. |
| LLM-as-judge (built-in) | `Faithfulness` | Hard gate: are the title/description claims supported by the source metadata + tool outputs? |
| Trajectory | `Tool Correctness`, `no_unsafe_write_action`, `required_tool_args`, `grounded_in_tool_output`, `no_excessive_tool_calls` | Scores agent *behavior*: right tool, no unsafe writes, required args present, grounded answer, no runaway loops. |
| Pairwise | `compare.py` | Head-to-head preference between two configs, randomized A/B order to reduce position bias. |
| Gate | `evaluators/gates.py` | Aggregates everything into `pass` / `review` / `fail`. |

## Package layout

```
evaluation/
  prompts.py            prompt variants (baseline = live system.md, strict_tools, schema_aware)
  config.py             model list, judge model, run-metadata tags
  models.py             LocalChatModel: wraps the agent's LLM as a DeepEval model
  target.py             runs the agent under one (prompt, model) config; extracts title/desc + trajectory
  extraction.py         pull title/description + tool trajectory from an agent turn
  metrics.py            DeepEval metrics (deterministic wrappers, GEval judges, faithfulness, tool correctness)
  dataset.py            starter suite (23 examples) of inputs + expected behavior
  compare.py            local pairwise comparator
  run_experiments.py    prompt x model matrix runner + gate + JSON/console report
  evaluators/
    deterministic.py    pure title/description checks
    trajectory.py       pure tool-behavior checks
    gates.py            pass/review/fail logic
```

## Prerequisites

DeepEval is installed (`pip install deepeval`). Set the agent's LLM env (the same
`.env` the app already uses):

```
OPENAI_API_KEY=...            # the agent's LLM endpoint key
OPENAI_BASE_URL=...           # e.g. the TACC endpoint
CKAN_LLM_MODEL=Meta-Llama-3.3-70B-Instruct

# Eval-specific (all optional):
CKAN_EVAL_MODELS=Meta-Llama-3.3-70B-Instruct   # comma-separated to sweep models
CKAN_EVAL_JUDGE_MODEL=Meta-Llama-3.3-70B-Instruct
```

No `LANGSMITH_API_KEY` is required — nothing leaves your endpoint. (DeepEval can
optionally push to a Confident AI dashboard if you `deepeval login`, but the runner
here writes local JSON reports and never requires it.)

## Quick start

```bash
# See the suite
python -m basic_ckan_agent.evaluation.dataset

# Run the full prompt x model matrix
python -m basic_ckan_agent.evaluation.run_experiments

# Run a single configuration
python -m basic_ckan_agent.evaluation.run_experiments --prompt baseline --model Meta-Llama-3.3-70B-Instruct

# Pairwise-compare two prompt variants on one model
python -m basic_ckan_agent.evaluation.run_experiments --pairwise baseline schema_aware
```

Each run prints a per-example `PASS/REVIEW/FAIL` line and a summary, and writes:

- a **self-contained HTML report** to `logs/eval/report-<timestamp>.html` — open
  it in any browser. It has the matrix summary, a colored pass/review/fail badge
  per example, the title/description scores, and a collapsible "details" panel per
  example with the generated text, gate reasons, and each judge's reason. No
  account, no network, nothing leaves your machine.
- a **JSON report** per config at `logs/eval/eval-<prompt>-<model>-<timestamp>.json`
  for diffing/automation, tagged with `prompt`, `model`, `graph_version` (git
  commit), `spec_version`, and `run_date`.

> DeepEval's own hosted UI is **Confident AI** (cloud, requires `deepeval login`).
> This suite deliberately uses a local HTML report instead so nothing leaves your
> endpoint. If you later want hosted history/charts, run `deepeval login`.

> **Throughput.** The agent and judges share the agent's global LLM rate limiter
> (`CKAN_LLM_REQUESTS_PER_SECOND`), so a full matrix is sequential and can take a
> while. Lower the suite (one `--prompt`/`--model`) while iterating.

## How-to

### Add a new eval example
Append a dict to `STARTER_EXAMPLES` in `dataset.py`:

```python
{
    "id": "title_my_new_case",
    "inputs": {
        "task_type": "metadata",          # "metadata" | "search" | "resources"
        "metadata": {"tags": [...], "spatial_coverage": "...", "resource_names": [...]},
        # or "question": "Find datasets about ..."
    },
    "outputs": {                           # expected *behavior*, not exact text
        "must_mention": ["term"],          # must appear in title+description
        "should_mention": ["term"],        # advisory; judged by the LLM
        "must_not": ["invent X"],          # advisory; judged by the LLM
        "expected_tools": ["package_search"],
        "forbidden_tools": ["package_update", ...],
        "required_args": {"package_search": ["payload_json"]},
        "write_approved": False,           # True skips the unsafe-write gate
        "max_tool_calls": 6,
    },
}
```

The runner turns each example into a DeepEval test case automatically and picks
the applicable metrics (metadata examples get the title/description judges +
faithfulness; tool examples get tool correctness + arg/grounding checks).

### Add a new prompt variant
Add `"my_variant": MY_PROMPT_TEXT` to `get_prompts()` in `prompts.py`. The matrix
picks it up automatically. `baseline` always tracks the live `system.md`.

### Add a new model
Set `CKAN_EVAL_MODELS=model_a,model_b`, or edit `_DEFAULT_MODELS` in `config.py`.
The endpoint must serve that model id.

### Add a new metric
- A code check: write `func(outputs, reference) -> {"key","score","comment"}` in
  `evaluators/`, wrap it with `FunctionMetric(func, "name")` in `metrics.py`, and
  add it to `metrics_for(...)`.
- An LLM judge: add a `GEval(...)` in `metrics.py` and map its score in
  `run_experiments._metric_to_feedback`.

### Vary node logic / tool descriptions
Those live in `runtime/graph.py` and `openapi/tools.py`. Change them on a branch
(the commit is captured as `graph_version`) and re-run — the score delta across
commits is your A/B.

## Interpreting pass / review / fail

Gate priority is **fail > review > pass** (`evaluators/gates.py`):

**PASS** — `title_basic_quality` and `description_basic_quality` true,
`faithfulness_pass` true, `title_score ≥ 3`, `description_score ≥ 3`, no unsafe
writes, required tool behavior satisfied.

**REVIEW** — `faithfulness_pass` false (but not a major fabrication),
`title_score ≤ 2`, `description_score ≤ 2`, expected tool not called, required
args missing, excessive tool calls, or missing must-mention terms. Also: a prompt
that **loses pairwise** to baseline on a meaningful share of examples.

**FAIL** — invents unsupported facts (faithfulness fails **and** the judge score
is very low → `major_issue`), performs a write during a read-only task,
empty/placeholder title or description, or an answer not grounded in tool outputs.

## Comparing experiments

- **Within a run:** the console matrix summary shows pass/review/fail per
  `(prompt, model)`.
- **Across runs:** diff the JSON reports in `logs/eval/` — each carries its full
  per-example feedback and the `graph_version`/`run_date` tags.
- **Head-to-head:** `--pairwise PROMPT_A PROMPT_B` runs both configs on every
  metadata example and tallies wins/ties with the faithfulness-first preference
  judge.

## Design notes

- **Your model does all the judging.** `LocalChatModel` (`models.py`) wraps the
  project's `build_model`, so GEval and faithfulness reuse the agent's client,
  rate limiter, and retries. DeepEval never supplies a model.
- **Non-native structured output.** Llama-class models answer in text, so
  `LocalChatModel.generate` parses the last JSON object and builds the pydantic
  schema GEval requests (tolerating extra keys).
- **Trajectory from outputs, not trace parsing.** The target captures tool calls
  into the test case's `tools_called` / `metadata`, keeping the trajectory checks
  pure and deterministic.
- **Faithfulness is a hard gate** in both the rubric judges and pairwise: a
  bland-but-accurate field outranks a polished invented one.
