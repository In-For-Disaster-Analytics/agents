from __future__ import annotations

import csv
import io
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from basic_ckan_agent.ckan.client import ckan_headers
from basic_ckan_agent.logging_config import LOG_FILE, logger
from basic_ckan_agent.runtime.graph import ChatSession
from basic_ckan_agent.settings import ckan_base_url, ckan_openapi_url

ESS_DIVE_TEST_DATASETS = Path(__file__).resolve().parents[3] / "tests" / "ess_dive_ckan_test_datasets.json"
AGENT_SMOKE_ANSWER_MAX_CHARS = 6000


@dataclass(frozen=True)
class EssDiveSmokeCase:
    name: str
    title: str
    initial_path: Path
    followup_paths: tuple[Path, Path]
    expected_fields: tuple[str, ...]


def smoke_test_ckan_search() -> None:
    logger.info("Running CKAN smoke tests")
    print("\nRunning CKAN smoke tests...")

    tests = [
        ("status_show", "status_show", {}),
        ("package_list", "package_list", {}),
        ("current_package_list_with_resources", "current_package_list_with_resources", {"limit": 5}),
        ("organization_list", "organization_list", {"all_fields": True, "include_dataset_count": True}),
        ("package_search all", "package_search", {"q": "*:*", "rows": 5}),
        ("package_search empty", "package_search", {"q": "", "rows": 5}),
        ("package_search flood", "package_search", {"q": "flood", "rows": 5}),
        ("package_search bethel", "package_search", {"q": "bethel", "rows": 5}),
        ("package_search Rising Up", "package_search", {"q": "Rising Up - Stories of the Flood", "rows": 5}),
    ]

    for label, action, payload in tests:
        url = f"{ckan_base_url()}/api/3/action/{action}"
        logger.info("SMOKE %s POST %s payload=%s", label, url, json.dumps(payload))

        print("\n" + "-" * 80)
        print(label)
        print(f"POST {url}")
        print(json.dumps(payload, indent=2))

        try:
            response = requests.post(url, headers=ckan_headers(), json=payload, timeout=30)
            print(f"HTTP {response.status_code}")
            _print_smoke_response(action, label, response)
        except Exception:
            logger.exception("Smoke test failed for %s", label)
            print(f"ERROR during {label}. See log file: {LOG_FILE}")


def smoke_test_ess_dive_agent_datasets() -> None:
    datasets = _load_ess_dive_smoke_datasets()
    if not datasets:
        print(f"No ESS-DIVE smoke datasets found at {ESS_DIVE_TEST_DATASETS}")
        return

    print("\nRunning ESS-DIVE agent smoke tests...")
    print(f"Fixture: {ESS_DIVE_TEST_DATASETS}")
    logger.info("Running ESS-DIVE agent smoke tests fixture=%s count=%s", ESS_DIVE_TEST_DATASETS, len(datasets))

    with tempfile.TemporaryDirectory(prefix="basic-ckan-agent-smoke-") as temp_dir:
        cases = _build_ess_dive_smoke_cases(datasets, Path(temp_dir))
        print(f"Staged ingestion files: {temp_dir}")

        for index, case in enumerate(cases, start=1):
            print("\n" + "=" * 80)
            print(f"ESS-DIVE dataset smoke {index}/{len(cases)}")
            print(f"{case.name} :: {case.title}")
            logger.info("SMOKE ESS-DIVE dataset=%s title=%s", case.name, case.title)

            session = ChatSession()
            turns = [
                ("initial package evidence", _initial_dataset_smoke_prompt(case, index, len(cases))),
                ("resource ingestion evidence", _resource_dataset_smoke_prompt(case)),
            ]
            for label, prompt in turns:
                print("\n" + "-" * 80)
                print(label)
                print(prompt)
                logger.info("SMOKE ESS-DIVE %s prompt=%s", label, prompt)
                try:
                    answer = session.ask(prompt)
                    _print_agent_smoke_answer(answer)
                except Exception:
                    logger.exception("ESS-DIVE agent smoke failed for %s during %s", case.name, label)
                    print(f"ERROR during {label}. See log file: {LOG_FILE}")
                    break


def _print_smoke_response(action: str, label: str, response: requests.Response) -> None:
    try:
        data = response.json()
    except ValueError:
        print(response.text)
        logger.exception("Smoke test returned non-JSON response")
        return

    logger.info(
        "SMOKE %s HTTP %s success=%s",
        label,
        response.status_code,
        data.get("success") if isinstance(data, dict) else None,
    )
    logger.debug("SMOKE RAW RESPONSE %s", json.dumps(data, indent=2, ensure_ascii=False, default=str))
    print(f"success: {data.get('success')}")

    if data.get("error"):
        print("error:")
        print(json.dumps(data["error"], indent=2))
        return

    result = data.get("result")
    if action == "status_show":
        print(json.dumps(result, indent=2)[:3000])
    elif action == "package_list":
        _print_package_list(result)
    elif action == "current_package_list_with_resources":
        _print_package_summaries("current packages returned", result)
    elif action == "organization_list":
        _print_organizations(result)
    elif action == "package_search":
        _print_package_search(result)
    else:
        print(json.dumps(result, indent=2)[:3000])


def _load_ess_dive_smoke_datasets(path: Path = ESS_DIVE_TEST_DATASETS) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    datasets = data.get("datasets")
    return datasets if isinstance(datasets, list) else []


def _build_ess_dive_smoke_cases(
    datasets: list[dict[str, Any]],
    stage_root: Path,
) -> list[EssDiveSmokeCase]:
    cases: list[EssDiveSmokeCase] = []
    stage_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        package = dataset.get("ckan_package") if isinstance(dataset, dict) else None
        if not isinstance(package, dict):
            continue

        name = str(package.get("name") or "ess-dive-dataset")
        title = str(package.get("title") or name)
        resources = package.get("resources") if isinstance(package.get("resources"), list) else []
        dataset_dir = stage_root / _slugify(name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        initial_path = dataset_dir / "ckan-package-fields.json"
        resource_csv_path = dataset_dir / "resources.csv"
        resource_notes_path = dataset_dir / "resource-notes.md"

        package_fields = {key: value for key, value in package.items() if key != "resources"}
        initial_path.write_text(
            json.dumps(
                {
                    "source": "ess_dive_ckan_test_datasets.json",
                    "purpose": "Expected non-resource CKAN package fields for the smoke-test dataset.",
                    "ckan_package": package_fields,
                    "resource_count": len(resources),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        resource_csv_path.write_text(_resources_csv(resources), encoding="utf-8")
        resource_notes_path.write_text(_resource_notes_markdown(package, resources), encoding="utf-8")

        cases.append(
            EssDiveSmokeCase(
                name=name,
                title=title,
                initial_path=initial_path,
                followup_paths=(resource_csv_path, resource_notes_path),
                expected_fields=tuple(package.keys()),
            )
        )

    return cases


def _resources_csv(resources: list[Any]) -> str:
    output = io.StringIO()
    fieldnames = ["index", "name", "description", "format", "mimetype", "url"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for index, resource in enumerate(resources, start=1):
        row = resource if isinstance(resource, dict) else {}
        writer.writerow(
            {
                "index": index,
                "name": row.get("name", ""),
                "description": row.get("description", ""),
                "format": row.get("format", ""),
                "mimetype": row.get("mimetype", ""),
                "url": row.get("url", ""),
            }
        )
    return output.getvalue()


def _resource_notes_markdown(package: dict[str, Any], resources: list[Any]) -> str:
    lines = [
        f"# {package.get('title') or package.get('name') or 'ESS-DIVE Dataset'}",
        "",
        str(package.get("notes") or "").strip(),
        "",
        "## Expected CKAN Resource Evidence",
        "",
    ]
    for index, resource in enumerate(resources, start=1):
        row = resource if isinstance(resource, dict) else {}
        lines.extend(
            [
                f"### Resource {index}: {row.get('name') or 'Unnamed resource'}",
                f"- Format: {row.get('format') or 'unknown'}",
                f"- MIME type: {row.get('mimetype') or 'unknown'}",
                f"- URL: {row.get('url') or 'unknown'}",
                f"- Description: {row.get('description') or ''}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _initial_dataset_smoke_prompt(case: EssDiveSmokeCase, index: int, total: int) -> str:
    expected_fields = ", ".join(field for field in case.expected_fields if field != "resources")
    return (
        f"ESS-DIVE agent smoke test {index}/{total}: {case.title}\n"
        f"Start with this one local file path: `{case.initial_path}`\n\n"
        "Use the local file tools before answering. Extract the expected non-resource CKAN package fields from "
        "that file and draft a concise metadata coverage checklist. Do not call CKAN write tools. "
        f"Fields to check: {expected_fields}."
    )


def _resource_dataset_smoke_prompt(case: EssDiveSmokeCase) -> str:
    resource_paths = "\n".join(f"- `{path}`" for path in case.followup_paths)
    return (
        "Continue the same ESS-DIVE smoke test. Now inspect these resource ingestion file paths:\n"
        f"{resource_paths}\n\n"
        "Use the local file tools to extract resource metadata, then update the CKAN draft coverage for the "
        "dataset. Include whether the resources field can be recovered, and call out any missing or uncertain "
        "package fields. Do not call CKAN write tools."
    )


def _print_agent_smoke_answer(answer: str) -> None:
    if len(answer) <= AGENT_SMOKE_ANSWER_MAX_CHARS:
        print(answer)
        return
    print(answer[:AGENT_SMOKE_ANSWER_MAX_CHARS])
    print(f"... truncated {len(answer) - AGENT_SMOKE_ANSWER_MAX_CHARS} characters")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug) or "dataset"


def _print_package_list(result: object) -> None:
    if isinstance(result, list):
        print(f"package_list count: {len(result)}")
        for name in result[:10]:
            print(f"- {name}")
        return
    print(json.dumps(result, indent=2)[:3000])


def _print_package_summaries(label: str, result: object) -> None:
    if isinstance(result, list):
        print(f"{label}: {len(result)}")
        for pkg in result[:5]:
            if isinstance(pkg, dict):
                print(f"- {pkg.get('name')} :: {pkg.get('title')}")
        return
    print(json.dumps(result, indent=2)[:3000])


def _print_organizations(result: object) -> None:
    if isinstance(result, list):
        print(f"organizations returned: {len(result)}")
        for org in result[:10]:
            if isinstance(org, dict):
                print(
                    f"- {org.get('name')} :: "
                    f"{org.get('title')} :: "
                    f"package_count={org.get('package_count')}"
                )
            else:
                print(f"- {org}")
        return
    print(json.dumps(result, indent=2)[:3000])


def _print_package_search(result: object) -> None:
    if isinstance(result, dict):
        print(f"count: {result.get('count')}")
        for item in result.get("results", [])[:5]:
            if isinstance(item, dict):
                print(f"- {item.get('name')} :: {item.get('title')}")
        return
    print(json.dumps(result, indent=2)[:3000])


def main() -> None:
    logger.info("CKAN_BASE_URL=%s", ckan_base_url())
    logger.info("CKAN_OPENAPI_URL=%s", ckan_openapi_url())
    logger.info("LOG_FILE=%s", LOG_FILE)

    print("\nCKAN OpenAPI LangGraph assistant")
    print(f"CKAN_BASE_URL={ckan_base_url()}")
    print(f"CKAN_OPENAPI_URL={ckan_openapi_url()}")
    print(f"Log file: {LOG_FILE}")
    print("Type q, quit, or exit to stop.")
    print("Type smoke to run ESS-DIVE agent dataset smoke tests.")
    print("Type smoke ckan for direct CKAN API smoke tests, or smoke all for both.\n")

    session = ChatSession()

    while True:
        try:
            user_input = input("User: ").strip()
        except EOFError:
            print()
            break

        if user_input.lower() in {"q", "quit", "exit"}:
            break
        command = user_input.lower()
        if command in {"smoke", "smoke datasets", "smoke ess-dive", "smoke ess dive"}:
            smoke_test_ess_dive_agent_datasets()
            continue
        if command in {"smoke ckan", "smoke api"}:
            smoke_test_ckan_search()
            continue
        if command == "smoke all":
            smoke_test_ckan_search()
            smoke_test_ess_dive_agent_datasets()
            continue

        try:
            print(f"Assistant: {session.ask(user_input)}")
        except Exception:
            logger.exception("Unhandled error while processing user input")
            print(f"Error occurred. See log file: {LOG_FILE}")
