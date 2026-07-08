# Basic CKAN Agent

Small CLI LangGraph agent that builds CKAN Action API tools from a live OpenAPI schema.

## Layout

```text
main.py                         # thin CLI entrypoint
basic_ckan_agent/
  runtime/
    cli.py                      # interactive shell and smoke tests
    graph.py                    # LangGraph wiring and per-turn orchestration
  llm/
    model.py                    # chat model construction
    router.py                   # action-selection prompt flow
    recovery.py                 # provider-specific response recovery
  session/
    memory.py                   # compact in-memory conversation state
    task_planning.py            # deterministic planning/recovery helpers
  files/                        # local file-inspection tools and planner catalog
  settings.py                   # env and project paths
  utils.py                      # generic JSON helpers
  prompts/basic_ckan/           # Markdown prompt templates
  openapi/                      # generic OpenAPI parsing/catalog helpers
  ckan/                         # CKAN-specific auth, URLs, tools, guardrails, compaction
```

OpenAPI helpers are intentionally generic. CKAN-specific behavior stays under `basic_ckan_agent/ckan/`, including action guardrails, CKAN Action API URLs, auth headers, and response compaction.

## Run

Use the parent `ckan-agent-api` environment because this CLI depends on LangGraph,
LangChain, Pydantic, Requests, and python-dotenv.

```bash
cd ckan-agent-api
conda env create -f environment.yml
conda activate ckan-agent-api
cd basic-ckan-agent
cp .env.sample .env
python main.py
```

Inside the interactive shell:

- `smoke` runs the ESS-DIVE fixture-driven agent smoke test for all datasets in `tests/ess_dive_ckan_test_datasets.json`.
- `smoke ckan` runs the direct CKAN Action API smoke checks.
- `smoke all` runs both suites.
