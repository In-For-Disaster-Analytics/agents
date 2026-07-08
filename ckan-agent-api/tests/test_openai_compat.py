import json

from fastapi.testclient import TestClient

from app.agents.ckan_registration.graph import get_runner
from app.agents.ckan_registration.schemas import AgentRunResponse, CkanRunRequest
from app.main import app

client = TestClient(app)


class FakeRunner:
    def invoke(self, request: object) -> AgentRunResponse:
        return AgentRunResponse(
            ok=True,
            thread_id="test-thread",
            command="analyze",
            status="complete",
            result={"ok": True, "message": "finished"},
        )


class RecordingRunner:
    def __init__(self) -> None:
        self.requests: list[CkanRunRequest] = []

    def invoke(self, request: CkanRunRequest) -> AgentRunResponse:
        self.requests.append(request)
        thread_id = request.session_id or "missing-thread"
        return AgentRunResponse(
            ok=True,
            thread_id=thread_id,
            command="dry-run",
            status="dry-run",
            result={"ok": True, "review_markdown": "## CKAN Dry Run\n- Ready"},
        )


class ReviewRunner:
    def invoke(self, request: object) -> AgentRunResponse:
        return AgentRunResponse(
            ok=True,
            thread_id="review-thread",
            command="analyze",
            status="analyzed",
            result={"ok": True, "review_markdown": "## Proposed CKAN Registration\n- Dataset name: demo"},
        )


class MetadataReportRunner:
    def __init__(self) -> None:
        self.requests: list[CkanRunRequest] = []

    def invoke(self, request: CkanRunRequest) -> AgentRunResponse:
        self.requests.append(request)
        return AgentRunResponse(
            ok=True,
            thread_id=request.session_id or "metadata-thread",
            command="metadata-report",
            status="metadata_report",
            result={"ok": True, "review_markdown": "## File Metadata\n\n## Best-Guess CKAN Starter Metadata"},
        )


class ExplodingRunner:
    def invoke(self, request: object) -> AgentRunResponse:
        raise AssertionError("runner should not be invoked")


class NeedsInputRunner:
    def invoke(self, request: object) -> AgentRunResponse:
        return AgentRunResponse(
            ok=False,
            thread_id="needs-input-thread",
            command="analyze",
            status="needs_input",
            result={
                "ok": False,
                "status": "needs_input",
                "review_markdown": "## More Dataset Input Needed\n\nPlease provide files.",
            },
            error="Please provide files.",
        )


def test_models_endpoint_supports_root_alias() -> None:
    canonical = client.get("/v1/models")
    alias = client.get("/models")

    assert canonical.status_code == 200
    assert alias.status_code == 200
    assert alias.json() == canonical.json()


def test_root_openai_aliases_are_hidden_from_openapi_schema() -> None:
    schema = client.get("/openapi.json").json()

    assert "/models" not in schema["paths"]
    assert "/chat/completions" not in schema["paths"]
    assert "/v1/models" in schema["paths"]
    assert "/v1/chat/completions" in schema["paths"]


def test_chat_completion_supports_streaming_response() -> None:
    app.dependency_overrides[get_runner] = lambda: FakeRunner()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "stream": True,
                "messages": [{"role": "user", "content": "Analyze this dataset"}],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"

    content_chunks = [
        json.loads(event)["choices"][0]["delta"]["content"] for event in events[:-1] if "content" in event
    ]
    assert "finished" in "".join(content_chunks)


def test_streaming_chat_greeting_returns_intro_without_creating_ckan_run() -> None:
    app.dependency_overrides[get_runner] = lambda: ExplodingRunner()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200

    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    content_chunks = [
        json.loads(event)["choices"][0]["delta"]["content"] for event in events[:-1] if "content" in event
    ]
    content = "".join(content_chunks)
    assert "file metadata report" in content
    assert "starter CKAN metadata guess" in content


def test_chat_completion_prefers_review_markdown_over_raw_json() -> None:
    app.dependency_overrides[get_runner] = lambda: ReviewRunner()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "messages": [{"role": "user", "content": "Analyze this dataset"}],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("## Next Options")
    assert "## Proposed CKAN Registration" in content
    assert "Thread ID: `review-thread`" in content
    assert '"result"' not in content


def test_chat_completion_reuses_prior_thread_id_for_dry_run() -> None:
    runner = RecordingRunner()
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "messages": [
                    {"role": "user", "content": "Analyze this dataset"},
                    {
                        "role": "assistant",
                        "content": "## Proposed CKAN Registration\n\nThread ID: `abc123`\nStatus: `analyzed`",
                    },
                    {"role": "user", "content": "Run the dry-run"},
                ],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    assert runner.requests[0].session_id == "abc123"
    context = runner.requests[0].model_dump().get("agent_context")
    assert context["interface"] == "openai_chat_compat"
    assert context["thread_id_from_history"] == "abc123"
    assert "## CKAN Dry Run" in response.json()["choices"][0]["message"]["content"]


def test_chat_completion_followup_uses_prior_metadata_context_instead_of_intro() -> None:
    runner = MetadataReportRunner()
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            "## File Metadata\n\n"
                            "## Best-Guess CKAN Starter Metadata\n"
                            "- Title: `Houston-Area Extensometer Compaction Campaign Folium Map`\n"
                            "- URL: `needs_user_input`\n\n"
                            "Thread ID: `metadata-thread-1`\n"
                            "Status: `metadata_report`"
                        ),
                    },
                    {"role": "user", "content": "Didn't the url get pulled from upstream?"},
                ],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    assert runner.requests[0].session_id == "metadata-thread-1"
    context = runner.requests[0].model_dump().get("conversation_context")
    assert context["has_prior_metadata_report"] is True
    assert "Didn't the url" in context["context_text"]
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("## Next Options")
    assert "## File Metadata" in content


def test_chat_completion_reuses_thread_id_from_raw_json_history() -> None:
    runner = RecordingRunner()
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "messages": [
                    {"role": "assistant", "content": '{"thread_id": "json-thread", "status": "analyzed"}'},
                    {"role": "user", "content": "dry run"},
                ],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    assert runner.requests[0].session_id == "json-thread"


def test_chat_completion_skips_missing_state_error_thread_id() -> None:
    runner = RecordingRunner()
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "messages": [
                    {"role": "assistant", "content": '{"thread_id": "good-thread", "status": "analyzed"}'},
                    {
                        "role": "assistant",
                        "content": (
                            '{"thread_id": "bad-thread", "status": "error", '
                            '"error": "No saved CKAN agent state found at '
                            '/tmp/ckan-agent-api/ckan-registration/bad-thread.json."}'
                        ),
                    },
                    {"role": "user", "content": "dry run"},
                ],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    assert runner.requests[0].session_id == "good-thread"


def test_chat_completion_dry_run_without_prior_thread_still_reports_metadata() -> None:
    runner = MetadataReportRunner()
    app.dependency_overrides[get_runner] = lambda: runner
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "messages": [{"role": "user", "content": "dry run"}],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("## Next Options")
    assert "## File Metadata" in content
    assert runner.requests[0].session_id is None


def test_chat_completion_needs_input_does_not_append_thread_id() -> None:
    app.dependency_overrides[get_runner] = lambda: NeedsInputRunner()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ckan-registration-agent",
                "messages": [{"role": "user", "content": "I don't have an existing dataset"}],
            },
        )
    finally:
        app.dependency_overrides.pop(get_runner, None)

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("## Next Options")
    assert "## More Dataset Input Needed" in content
    assert "Thread ID" not in content
