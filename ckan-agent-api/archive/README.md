# Archive

Superseded code retained for reference (Increment 5 of the conversational
multi-persona metadata agent — see `docs/design/2026-06-26-conversational-metadata-persona-agent.md`).

Nothing here is imported by the live `app/` package. These were moved (not deleted)
so they can be restored if needed.

## Contents

- `basic-ckan-agent/` — the standalone basic CKAN agent. Its file extractors were
  migrated into `app/files/extractors/` (Increment 4); its LLM tool-calling layer
  (`files/tools.py`, `files/catalog.py`, `files/tool_catalog.yaml`, `files/schemas.py`)
  and its evaluation harness (`evaluation/`) are not used by the persona-chat agent and
  are archived here per the 2026-06-26 decision.
- `tests/test_basic_ckan_file_tools.py`, `tests/test_basic_ckan_smoke.py` — tests that
  import `basic_ckan_agent`; moved out of the live `tests/` suite alongside the agent.

## Not yet archived

The single-pass metadata path (`app/prompts/ckan_registration/metadata_guide.md` and the
`build_file_metadata_report` / `_prompt_guided_metadata` / `_guess_dataset_metadata`
helpers in `app/agents/ckan_registration/nodes.py`) is still the live default while
`CKAN_PERSONA_CHAT` is off. It will be archived only once the persona-chat path becomes
the default.
