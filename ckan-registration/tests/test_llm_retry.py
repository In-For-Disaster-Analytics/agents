"""Tests for LLM retry-with-exponential-backoff in _chat_completion_content.

Mirrors the mocking style used in test_ckan_throttle.py.

All HTTP is mocked; no live network calls are made.  The OpenAI SDK is either
absent (HTTP-fallback tests) or replaced by a fake client.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure src/ is on path so gam_registration package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gam_registration.utils as utils  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http_response(status_code: int, body: dict | None = None, headers: dict | None = None):
    """Return a MagicMock that looks like a requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if status_code >= 400:
        import requests as _req
        resp.raise_for_status.side_effect = _req.HTTPError(
            response=resp, request=None
        )
    else:
        resp.raise_for_status.return_value = None
    if body is not None:
        resp.json.return_value = body
    return resp


def _ok_llm_response(content: str = "result-text"):
    """200 OK response with a minimal LLM choices payload."""
    return _http_response(
        200,
        body={"choices": [{"message": {"content": content}}]},
    )


def _call_http(monkeypatch, **kwargs):
    """Call _chat_completion_content with OpenAI forced to None (HTTP branch)."""
    monkeypatch.setattr(utils, "OpenAI", None)
    return utils._chat_completion_content(
        model="test-model",
        api_key="test-key",
        system_prompt="sys",
        user_payload={"x": 1},
        base_url="https://llm.example.com",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Config-helper unit tests
# ---------------------------------------------------------------------------

def test_llm_max_retries_default(monkeypatch):
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    assert utils._llm_max_retries() == 5


def test_llm_max_retries_env(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    assert utils._llm_max_retries() == 3


def test_llm_max_retries_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "not-a-number")
    assert utils._llm_max_retries() == 5


def test_llm_max_retries_zero(monkeypatch):
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    assert utils._llm_max_retries() == 0


def test_llm_backoff_base_default(monkeypatch):
    monkeypatch.delenv("LLM_BACKOFF_BASE_SECONDS", raising=False)
    assert utils._llm_backoff_base_seconds() == 2.0


def test_llm_backoff_base_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "1.5")
    assert utils._llm_backoff_base_seconds() == 1.5


def test_llm_backoff_base_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "bad")
    assert utils._llm_backoff_base_seconds() == 2.0


def test_llm_backoff_max_default(monkeypatch):
    monkeypatch.delenv("LLM_BACKOFF_MAX_SECONDS", raising=False)
    assert utils._llm_backoff_max_seconds() == 60.0


def test_llm_backoff_max_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "30.0")
    assert utils._llm_backoff_max_seconds() == 30.0


def test_llm_backoff_max_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "bad")
    assert utils._llm_backoff_max_seconds() == 60.0


# ---------------------------------------------------------------------------
# HTTP branch: two 429s then success -> retried, content returned
# ---------------------------------------------------------------------------

def test_http_retries_429_twice_then_succeeds(monkeypatch):
    """429 twice → 200: call succeeds, sleep called exactly twice."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "1.0")

    side_effects = [
        _http_response(429),
        _http_response(429),
        _ok_llm_response("hello"),
    ]

    with patch.object(utils, "OpenAI", None), \
         patch.object(utils.requests, "post", side_effect=side_effects) as mock_post, \
         patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.random, "uniform", return_value=0.0):
        result = utils._chat_completion_content(
            model="m",
            api_key="k",
            system_prompt="s",
            user_payload={},
            base_url="https://llm.example.com",
        )

    assert result == "hello"
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# HTTP branch: all 429s -> raises after max_retries+1 attempts
# ---------------------------------------------------------------------------

def test_http_exhausts_retries_and_raises(monkeypatch):
    """All attempts 429 → raises HTTPError after LLM_MAX_RETRIES+1 total calls."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "1.0")

    import requests as _req

    with patch.object(utils, "OpenAI", None), \
         patch.object(utils.requests, "post", return_value=_http_response(429)) as mock_post, \
         patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.random, "uniform", return_value=0.0):
        try:
            utils._chat_completion_content(
                model="m",
                api_key="k",
                system_prompt="s",
                user_payload={},
                base_url="https://llm.example.com",
            )
            assert False, "Expected HTTPError to be raised"
        except _req.HTTPError:
            pass

    # max_retries=2 → 3 total attempts, 2 sleeps
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# HTTP branch: non-retryable 400 -> raises immediately, no sleep
# ---------------------------------------------------------------------------

def test_http_non_retryable_400_raises_immediately(monkeypatch):
    """400 is not in retryable set → raise_for_status immediately, no retry."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "1.0")

    import requests as _req

    with patch.object(utils, "OpenAI", None), \
         patch.object(utils.requests, "post", return_value=_http_response(400)) as mock_post, \
         patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.random, "uniform", return_value=0.0):
        try:
            utils._chat_completion_content(
                model="m",
                api_key="k",
                system_prompt="s",
                user_payload={},
                base_url="https://llm.example.com",
            )
            assert False, "Expected HTTPError"
        except _req.HTTPError:
            pass

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# HTTP branch: Retry-After header is honored
# ---------------------------------------------------------------------------

def test_http_retry_after_header_honored(monkeypatch):
    """If Retry-After: 2 header is present, sleep is called with ~2 (plus jitter)."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "60.0")

    resp_429 = _http_response(429, headers={"Retry-After": "2"})

    with patch.object(utils, "OpenAI", None), \
         patch.object(utils.requests, "post", side_effect=[resp_429, _ok_llm_response()]) as mock_post, \
         patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.random, "uniform", return_value=0.0):
        utils._chat_completion_content(
            model="m",
            api_key="k",
            system_prompt="s",
            user_payload={},
            base_url="https://llm.example.com",
        )

    assert mock_sleep.call_count == 1
    actual_delay = mock_sleep.call_args[0][0]
    # With jitter=0 the delay should equal float(Retry-After header) = 2.0
    assert actual_delay == 2.0, f"Expected 2.0, got {actual_delay}"


# ---------------------------------------------------------------------------
# HTTP branch: Retry-After header capped at LLM_BACKOFF_MAX_SECONDS
# ---------------------------------------------------------------------------

def test_http_retry_after_capped_at_max(monkeypatch):
    """Retry-After value larger than max is capped."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "5.0")

    resp_429 = _http_response(429, headers={"Retry-After": "999"})

    with patch.object(utils, "OpenAI", None), \
         patch.object(utils.requests, "post", side_effect=[resp_429, _ok_llm_response()]), \
         patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.random, "uniform", return_value=0.0):
        utils._chat_completion_content(
            model="m",
            api_key="k",
            system_prompt="s",
            user_payload={},
            base_url="https://llm.example.com",
        )

    actual_delay = mock_sleep.call_args[0][0]
    assert actual_delay <= 5.0, f"Expected delay capped at 5.0, got {actual_delay}"


# ---------------------------------------------------------------------------
# SDK branch: fake client raises retryable error twice then succeeds
# ---------------------------------------------------------------------------

def test_sdk_retries_on_rate_limit_error(monkeypatch):
    """SDK branch: RateLimitError twice then success -> retried, content returned."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "1.0")

    # Build a fake RateLimitError with status_code=429.
    class FakeRateLimitError(Exception):
        status_code = 429

    # Build a fake successful completion response.
    fake_choice = MagicMock()
    fake_choice.message.content = "sdk-content"
    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise FakeRateLimitError("rate limited")
        return fake_completion

    fake_chat_completions = MagicMock()
    fake_chat_completions.create.side_effect = fake_create

    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions = fake_chat_completions

    FakeOpenAI = MagicMock(return_value=fake_openai_client)

    with patch.object(utils, "OpenAI", FakeOpenAI), \
         patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.random, "uniform", return_value=0.0):
        result = utils._chat_completion_content(
            model="m",
            api_key="k",
            system_prompt="s",
            user_payload={},
            base_url="https://llm.example.com",
        )

    assert result == "sdk-content"
    assert call_count["n"] == 3
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# SDK branch: non-retryable exception class raises immediately
# ---------------------------------------------------------------------------

def test_sdk_non_retryable_exception_raises_immediately(monkeypatch):
    """SDK branch: unknown exception class -> not retried, raises immediately."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_BACKOFF_BASE_SECONDS", "0.01")
    monkeypatch.setenv("LLM_BACKOFF_MAX_SECONDS", "1.0")

    class SomeOtherError(Exception):
        pass

    def fake_create(**kwargs):
        raise SomeOtherError("unexpected")

    fake_chat_completions = MagicMock()
    fake_chat_completions.create.side_effect = fake_create
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions = fake_chat_completions
    FakeOpenAI = MagicMock(return_value=fake_openai_client)

    with patch.object(utils, "OpenAI", FakeOpenAI), \
         patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.random, "uniform", return_value=0.0):
        try:
            utils._chat_completion_content(
                model="m",
                api_key="k",
                system_prompt="s",
                user_payload={},
                base_url="https://llm.example.com",
            )
            assert False, "Expected SomeOtherError"
        except SomeOtherError:
            pass

    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# SDK branch: SDK's own max_retries is disabled (max_retries=0 passed)
# ---------------------------------------------------------------------------

def test_sdk_client_created_with_max_retries_zero(monkeypatch):
    """OpenAI client must be constructed with max_retries=0."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")

    fake_choice = MagicMock()
    fake_choice.message.content = "ok"
    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]

    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.return_value = fake_completion

    FakeOpenAI = MagicMock(return_value=fake_openai_client)

    with patch.object(utils, "OpenAI", FakeOpenAI):
        utils._chat_completion_content(
            model="m",
            api_key="mykey",
            system_prompt="s",
            user_payload={},
            base_url="https://llm.example.com",
        )

    _, kwargs = FakeOpenAI.call_args
    assert kwargs.get("max_retries") == 0, (
        f"Expected max_retries=0 but got {kwargs.get('max_retries')}"
    )


# ---------------------------------------------------------------------------
# HTTP branch: first call succeeds -> no retry, no sleep
# ---------------------------------------------------------------------------

def test_http_success_first_try_no_sleep(monkeypatch):
    """A 200 on the first attempt means no sleep is called."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")

    with patch.object(utils, "OpenAI", None), \
         patch.object(utils.requests, "post", return_value=_ok_llm_response("direct")) as mock_post, \
         patch.object(utils.time, "sleep") as mock_sleep:
        result = utils._chat_completion_content(
            model="m",
            api_key="k",
            system_prompt="s",
            user_payload={},
            base_url="https://llm.example.com",
        )

    assert result == "direct"
    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()
