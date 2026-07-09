from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.ckan_registration import nodes
from app.agents.ckan_registration.graph import CkanRegistrationRunner
from app.agents.ckan_registration.schemas import CkanRunRequest
from app.settings import Settings


def test_openai_chat_model_uses_settings(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def invoke(self, messages: list[tuple[str, str]]) -> Any:
            assert messages == [("system", "Classify the action."), ("human", "Please dry run this dataset.")]
            return type("Response", (), {"content": "dry-run"})()

    # _invoke_openai_chat delegates to app.llm.invoke_chat, so patch the client there.
    monkeypatch.setattr("app.llm.ChatOpenAI", FakeChatOpenAI)

    content = nodes._invoke_openai_chat(
        Settings(
            openai_base_url="https://api.openai.test/v1",
            openai_api_key="sk-test",
            ckan_llm_model="gpt-test",
        ),
        [
            {"role": "system", "content": "Classify the action."},
            {"role": "user", "content": "Please dry run this dataset."},
        ],
        temperature=0.2,
        max_tokens=12,
        timeout=15,
    )

    assert content == "dry-run"
    assert calls == [
        {
            "model": "gpt-test",
            "api_key": "sk-test",
            "temperature": 0.2,
            "max_tokens": 12,
            "base_url": "https://api.openai.test/v1",
        }
    ]


def test_runner_reads_file_and_returns_metadata_report(tmp_path: Path) -> None:
    data_file = tmp_path / "folium_mapping.txt"
    data_file.write_text(
        "# Folium Mapping\n\n"
        "import folium\n"
        "import pandas as pd\n"
        "This notebook output describes an interactive geospatial map.",
        encoding="utf-8",
    )
    settings = Settings(state_dir=tmp_path / "state", checkpoint_db=tmp_path / "checkpoints.sqlite")
    runner = CkanRegistrationRunner(settings)

    response = runner.invoke(
        CkanRunRequest(
            message="Help me create CKAN metadata for this dataset.",
            files=[str(data_file)],
        )
    )

    assert response.ok is True
    assert response.command == "metadata-report"
    assert response.status == "metadata_report"
    assert response.requires_action is None
    assert response.result["readable_file_count"] == 1
    assert response.result["metadata_guess"]["title"] == "Folium Mapping"
    assert "folium" in response.result["metadata_guess"]["tags"]
    assert "To analyze, I need to know where your data is" not in response.result["review_markdown"]
    assert "Ready to validate this with CKAN?" in response.result["review_markdown"]
    assert "Is this a new CKAN dataset" in response.result["review_markdown"]

    state_path = Path(response.result["state_path"])
    assert state_path.exists()
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["status"] == "metadata_report"
    assert saved_state["desired_dataset_payload"]["title"] == "Folium Mapping"
    assert saved_state["resource_plan"][0]["local_path"] == str(data_file)


def test_runner_starts_from_filename_hint_without_clarification(tmp_path: Path) -> None:
    settings = Settings(state_dir=tmp_path / "state", checkpoint_db=tmp_path / "checkpoints.sqlite")
    runner = CkanRegistrationRunner(settings)

    response = runner.invoke(
        CkanRunRequest(
            message="I uploaded folium_mapping.txt and need CKAN metadata.",
        )
    )

    assert response.ok is True
    assert response.status == "metadata_report"
    assert response.requires_action is None
    assert response.result["files"][0]["name"] == "folium_mapping.txt"
    assert response.result["metadata_guess"]["title"] == "Folium Mapping"
    assert "To analyze, I need to know where your data is" not in response.result["review_markdown"]


def test_metadata_guide_prompt_can_shape_metadata(monkeypatch: Any, tmp_path: Path) -> None:
    data_file = tmp_path / "folium_mapping.txt"
    data_file.write_text("# Source Evidence\n\nfolium map output", encoding="utf-8")

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            assert "Core Rules" in kwargs["messages"][0]["content"]
            assert "Didn't the url get pulled from upstream?" in kwargs["messages"][1]["content"]
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": """
{
  "status": "metadata_report",
  "confidence": "medium",
  "ckan_package": {
    "title": "Prompt Guided Folium Map",
    "name": "prompt-guided-folium-map",
    "notes": "Prompt-guided notes from the metadata guide.",
    "tags": ["folium", "prompted"]
  },
  "resources": [
    {
      "name": "folium-map",
      "description": "Prompt-guided resource description.",
      "format": "TXT",
      "mimetype": "text/plain",
      "resource_type": "visualization",
      "upload_recommendation": "upload",
      "reason": "The supplied file is readable."
    }
  ],
  "needs_user_input": [
    {
      "field": "author",
      "question": "What should the author be?",
      "why_needed": "The parsed file does not contain an author.",
      "example": "Data Services Team"
    }
  ],
  "evidence_summary": ["Readable folium mapping text file."],
  "warnings": []
}
"""
                                    },
                                )(),
                            },
                        )()
                    ]
                },
            )()

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("app.llm.ChatOpenAI", None)
    monkeypatch.setattr("app.llm.OpenAI", FakeOpenAI)
    settings = Settings(
        state_dir=tmp_path / "state",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        openai_api_key="test-key",
    )
    runner = CkanRegistrationRunner(settings)

    response = runner.invoke(
        CkanRunRequest(
            message="Use the guide for CKAN metadata.",
            files=[str(data_file)],
            conversation_context={
                "context_text": (
                    "assistant: ## Best-Guess CKAN Starter Metadata\n"
                    "- URL: `needs_user_input`\n"
                    "user: Didn't the url get pulled from upstream?"
                ),
                "has_prior_metadata_report": True,
            },
        )
    )

    assert response.ok is True
    assert response.result["metadata_prompt"]["used"] is True
    assert response.result["metadata_guess"]["title"] == "Prompt Guided Folium Map"
    assert response.result["metadata_guess"]["name"] == "prompt-guided-folium-map"
    assert response.result["prompt_metadata"]["needs_user_input"][0]["field"] == "author"
    assert "What should the author be?" in response.result["review_markdown"]


def test_prompt_package_url_is_visible_in_report(monkeypatch: Any, tmp_path: Path) -> None:
    data_file = tmp_path / "popup.html"
    data_file.write_text("fetch('https://upstream.example.org/stations/123/measurements.json')", encoding="utf-8")

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": """
{
  "status": "metadata_report",
  "confidence": "medium",
  "ckan_package": {
    "title": "Upstream URL Map",
    "name": "upstream-url-map",
    "notes": "Uses explicit Upstream JSON URLs found in the file.",
    "url": "https://upstream.example.org/stations/123/measurements.json",
    "tags": ["upstream"]
  },
  "resources": [],
  "needs_user_input": [],
  "evidence_summary": ["The URL appears in popup.html."],
  "warnings": []
}
"""
                                    },
                                )(),
                            },
                        )()
                    ]
                },
            )()

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("app.llm.ChatOpenAI", None)
    monkeypatch.setattr("app.llm.OpenAI", FakeOpenAI)
    settings = Settings(
        state_dir=tmp_path / "state",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        openai_api_key="test-key",
    )
    runner = CkanRegistrationRunner(settings)

    response = runner.invoke(
        CkanRunRequest(
            message="Didn't the url get pulled from upstream?",
            files=[str(data_file)],
        )
    )

    assert response.result["metadata_guess"]["url"] == "https://upstream.example.org/stations/123/measurements.json"
    assert "- Url: `https://upstream.example.org/stations/123/measurements.json`" in response.result["review_markdown"]


def test_dry_run_request_routes_to_registration_path(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeLegacyWorker:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings
            pass

        def run(self, command: str, request: dict[str, Any]) -> dict[str, Any]:
            calls.append((command, request))
            state_path = self.settings.state_dir / f"{request['session_id']}.json"
            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            assert saved_state["desired_dataset_payload"]["owner_org"] == "2f68b69f-95b8-468c-b0c0-39d916f26c61"
            assert saved_state["ckan"]["owner_org_label"] == "DSO-Institute"
            return {
                "ok": True,
                "command": command,
                "status": "dry_run",
                "review_markdown": "## CKAN Dry Run\n- Ready",
            }

    class FakeCkanClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def resolve_organization_id(self, organization: str) -> dict[str, str]:
            assert organization == "DSO-Institute"
            return {
                "id": "2f68b69f-95b8-468c-b0c0-39d916f26c61",
                "name": "dso-institute",
                "title": "DSO-Institute",
                "matched_by": "organization_list",
            }

    monkeypatch.setattr(nodes, "LegacyCkanWorker", FakeLegacyWorker)
    monkeypatch.setattr(nodes, "CkanClient", FakeCkanClient)
    settings = Settings(state_dir=tmp_path / "state", checkpoint_db=tmp_path / "checkpoints.sqlite")
    runner = CkanRegistrationRunner(settings)

    metadata_response = runner.invoke(
        CkanRunRequest(
            session_id="dry-run-thread",
            message="Create metadata for a new dataset.",
            files=[str(tmp_path / "missing.txt")],
        )
    )
    assert metadata_response.status == "metadata_report"

    response = runner.invoke(CkanRunRequest(session_id="dry-run-thread", message="please run a dry run"))

    assert response.ok is True
    assert response.command == "dry-run"
    assert calls[0][0] == "dry-run"


def test_register_before_dry_run_is_blocked(tmp_path: Path) -> None:
    data_file = tmp_path / "folium_mapping.txt"
    data_file.write_text("# Folium Mapping\n\ncontent", encoding="utf-8")
    settings = Settings(state_dir=tmp_path / "state", checkpoint_db=tmp_path / "checkpoints.sqlite")
    runner = CkanRegistrationRunner(settings)

    metadata_response = runner.invoke(
        CkanRunRequest(
            message="Create metadata.",
            files=[str(data_file)],
        )
    )
    apply_response = runner.invoke(
        CkanRunRequest(
            session_id=metadata_response.thread_id,
            message="REGISTER",
        )
    )

    assert apply_response.ok is False
    assert apply_response.status == "needs_dry_run"
    assert "Run a CKAN dry-run before registering" in apply_response.result["review_markdown"]


def test_register_response_reports_uploaded_resources(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeLegacyWorker:
        def __init__(self, settings: Settings) -> None:
            pass

        def run(self, command: str, request: dict[str, Any]) -> dict[str, Any]:
            assert command == "apply"
            return {
                "ok": True,
                "command": "apply",
                "status": "applied",
                "dataset_name": "demo-dataset",
                "dataset_url": "https://ckan.example/dataset/demo-dataset",
                "resource_count": 2,
                "resource_created": 1,
                "resource_updated": 1,
                "resource_removed": 0,
            }

    class FakeCkanClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def resolve_organization_id(self, organization: str) -> dict[str, str]:
            return {"id": "org-id", "name": "dso-institute", "title": "DSO-Institute", "matched_by": "test"}

    monkeypatch.setattr(nodes, "LegacyCkanWorker", FakeLegacyWorker)
    monkeypatch.setattr(nodes, "CkanClient", FakeCkanClient)
    data_file = tmp_path / "folium_mapping.txt"
    data_file.write_text("# Folium Mapping\n\ncontent", encoding="utf-8")
    settings = Settings(state_dir=tmp_path / "state", checkpoint_db=tmp_path / "checkpoints.sqlite")
    runner = CkanRegistrationRunner(settings)

    metadata_response = runner.invoke(
        CkanRunRequest(
            message="Create metadata for a new dataset.",
            files=[str(data_file)],
        )
    )
    state_path = Path(metadata_response.result["state_path"])
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    saved_state["status"] = "dry_run"
    state_path.write_text(json.dumps(saved_state), encoding="utf-8")

    apply_response = runner.invoke(CkanRunRequest(session_id=metadata_response.thread_id, message="REGISTER"))

    assert apply_response.ok is True
    assert apply_response.status == "applied"
    assert "Resources uploaded/sent to CKAN: `2`" in apply_response.result["review_markdown"]
    assert "`1` created, `1` updated, `0` removed" in apply_response.result["review_markdown"]


def test_dataset_intent_reply_resumes_dry_run() -> None:
    """A 'new dataset' reply after a needs_dataset_intent prompt must resume the dry-run
    (carrying the intent), not be misclassified as a fresh analyze (regression for the
    'dry run → new dataset → REGISTER refused' bug)."""
    intake = nodes.make_intake_node()

    out = intake(
        {
            "thread_id": "t1",
            "action": "",
            "status": "needs_dataset_intent",
            "request": {"session_id": "t1", "message": "new dataset"},
        }
    )
    assert out["action"] == "dry-run"
    assert out["request"]["dataset_intent"] == "new"

    # "update <name>" likewise resumes the dry-run as an update.
    out_update = intake(
        {
            "thread_id": "t2",
            "action": "",
            "status": "needs_dataset_intent",
            "request": {"session_id": "t2", "message": "update houston-extensometer-map"},
        }
    )
    assert out_update["action"] == "dry-run"
    assert out_update["request"]["dataset_intent"] == "update"


def test_new_dataset_reply_without_prior_intent_prompt_is_analyze() -> None:
    """Outside the needs_dataset_intent context, 'new dataset' keeps the old analyze default."""
    intake = nodes.make_intake_node()
    out = intake(
        {
            "thread_id": "t3",
            "action": "",
            "status": "analyzed",
            "request": {"session_id": "t3", "message": "new dataset"},
        }
    )
    assert out["action"] == "analyze"


def test_ambiguous_post_analysis_message_is_revise() -> None:
    """After analysis, messages like 'zoom in to the title' should be revise, not re-analyze."""
    intake = nodes.make_intake_node()
    for msg in [
        "can we zoom in to the location in the title?",
        "the title should be more specific",
        "willy style",
        "change the location to street level",
    ]:
        out = intake(
            {
                "thread_id": "t-revise",
                "action": "",
                "status": "analyzed",
                "request": {"session_id": "t-revise", "message": msg},
            }
        )
        assert out["action"] == "revise", f"expected revise for {msg!r}, got {out['action']!r}"


def test_validate_without_new_or_update_choice_asks_for_dataset_intent(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeCkanClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def package_search(self, query: str, *, rows: int = 10) -> list[dict[str, Any]]:
            return [
                {
                    "id": "existing-1-id",
                    "name": "houston-extensometer-map",
                    "title": "Houston Extensometer Map",
                    "owner_org": "org-id",
                }
            ]

    monkeypatch.setattr(nodes, "CkanClient", FakeCkanClient)
    data_file = tmp_path / "folium_mapping.txt"
    data_file.write_text("# Houston Extensometer Map\n\ncontent", encoding="utf-8")
    settings = Settings(state_dir=tmp_path / "state", checkpoint_db=tmp_path / "checkpoints.sqlite")
    runner = CkanRegistrationRunner(settings)

    metadata_response = runner.invoke(
        CkanRunRequest(
            message="Create metadata.",
            files=[str(data_file)],
        )
    )
    validate_response = runner.invoke(
        CkanRunRequest(
            session_id=metadata_response.thread_id,
            message="validate",
        )
    )

    assert validate_response.ok is False
    assert validate_response.status == "needs_dataset_intent"
    assert "new CKAN dataset" in validate_response.result["review_markdown"]
    assert validate_response.result["candidate_existing_datasets"][0]["name"] == "houston-extensometer-map"


def test_update_without_target_and_multiple_matches_asks_which_dataset(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    class FakeCkanClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def package_search(self, query: str, *, rows: int = 10) -> list[dict[str, Any]]:
            return [
                {"id": "a", "name": "houston-extensometer-map", "title": "Houston Extensometer Map"},
                {"id": "b", "name": "houston-extensometer-campaign", "title": "Houston Extensometer Campaign"},
            ]

    monkeypatch.setattr(nodes, "CkanClient", FakeCkanClient)
    data_file = tmp_path / "folium_mapping.txt"
    data_file.write_text("# Houston Extensometer Map\n\ncontent", encoding="utf-8")
    settings = Settings(state_dir=tmp_path / "state", checkpoint_db=tmp_path / "checkpoints.sqlite")
    runner = CkanRegistrationRunner(settings)

    metadata_response = runner.invoke(
        CkanRunRequest(
            message="Create metadata.",
            files=[str(data_file)],
        )
    )
    update_response = runner.invoke(
        CkanRunRequest(
            session_id=metadata_response.thread_id,
            message="This is updating an existing dataset. validate",
        )
    )

    assert update_response.ok is False
    assert update_response.status == "needs_existing_dataset_choice"
    assert "update <dataset-name>" in update_response.result["review_markdown"]
    assert len(update_response.result["candidate_existing_datasets"]) == 2
