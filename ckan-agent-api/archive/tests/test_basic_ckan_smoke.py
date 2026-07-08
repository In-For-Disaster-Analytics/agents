from __future__ import annotations

import json
import sys
from pathlib import Path

BASIC_AGENT_ROOT = Path(__file__).resolve().parents[1] / "basic-ckan-agent"
sys.path.insert(0, str(BASIC_AGENT_ROOT))

from basic_ckan_agent.files.safety import extract_file_paths  # noqa: E402
from basic_ckan_agent.runtime.cli import (  # noqa: E402
    _build_ess_dive_smoke_cases,
    _initial_dataset_smoke_prompt,
    _load_ess_dive_smoke_datasets,
    _resource_dataset_smoke_prompt,
)


def test_ess_dive_smoke_cases_stage_one_initial_path_then_two_resource_paths(tmp_path: Path) -> None:
    datasets = _load_ess_dive_smoke_datasets()

    cases = _build_ess_dive_smoke_cases(datasets, tmp_path)

    assert len(cases) == 5
    for index, case in enumerate(cases, start=1):
        assert case.initial_path.exists()
        assert len(case.followup_paths) == 2
        assert all(path.exists() for path in case.followup_paths)

        overview = json.loads(case.initial_path.read_text(encoding="utf-8"))
        assert "ckan_package" in overview
        assert "resources" not in overview["ckan_package"]
        assert overview["resource_count"] > 0

        initial_prompt = _initial_dataset_smoke_prompt(case, index, len(cases))
        resource_prompt = _resource_dataset_smoke_prompt(case)

        assert extract_file_paths(initial_prompt) == [str(case.initial_path)]
        assert extract_file_paths(resource_prompt) == [str(path) for path in case.followup_paths]
        assert "resources" in case.expected_fields


def test_ess_dive_resource_smoke_files_include_all_fixture_resources(tmp_path: Path) -> None:
    datasets = _load_ess_dive_smoke_datasets()
    cases = _build_ess_dive_smoke_cases(datasets, tmp_path)

    for dataset, case in zip(datasets, cases, strict=True):
        package = dataset["ckan_package"]
        resources = package["resources"]
        resource_csv = case.followup_paths[0].read_text(encoding="utf-8")
        resource_notes = case.followup_paths[1].read_text(encoding="utf-8")

        for resource in resources:
            assert resource["name"] in resource_csv
            assert resource["name"] in resource_notes
            assert resource["url"] in resource_csv
