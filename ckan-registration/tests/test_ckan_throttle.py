"""Tests for the CKAN inter-call throttle (CKAN_CALL_DELAY_SECONDS).

Verifies ckan_action_post sleeps before each request, that the delay is
configurable via the env var, and that 0 disables it. All HTTP is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is on path so gam_registration package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gam_registration.utils as utils  # noqa: E402


def _ok_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"success": True, "result": {"id": "x"}}
    return resp


def test_delay_default_is_half_second(monkeypatch):
    monkeypatch.delenv("CKAN_CALL_DELAY_SECONDS", raising=False)
    assert utils._ckan_call_delay_seconds() == 0.5


def test_delay_reads_env(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "2.5")
    assert utils._ckan_call_delay_seconds() == 2.5


def test_delay_zero_disables(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "0")
    assert utils._ckan_call_delay_seconds() == 0.0


def test_delay_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "not-a-number")
    assert utils._ckan_call_delay_seconds() == 0.5


def test_ckan_action_post_sleeps_before_request(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "1.5")
    with patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.requests, "post", return_value=_ok_response()) as mock_post:
        result = utils.ckan_action_post(
            "https://ckan.example.org", "resource_create", {"a": 1}, "Bearer t"
        )
    mock_sleep.assert_called_once_with(1.5)
    assert mock_post.called
    assert result == {"id": "x"}


def test_ckan_action_post_no_sleep_when_zero(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "0")
    with patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.requests, "post", return_value=_ok_response()):
        utils.ckan_action_post(
            "https://ckan.example.org", "package_show", {"id": "x"}, None
        )
    mock_sleep.assert_not_called()


def test_throttle_applies_per_call_across_many_uploads(monkeypatch):
    # N resource uploads -> N sleeps (one before each call).
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "0.3")
    with patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.requests, "post", return_value=_ok_response()):
        for _ in range(5):
            utils.ckan_action_post(
                "https://ckan.example.org", "resource_create", {"a": 1}, "Bearer t"
            )
    assert mock_sleep.call_count == 5
    assert all(c.args == (0.3,) for c in mock_sleep.call_args_list)


# ===========================================================================
# CKAN retry on transient gateway errors (502/503/504/500/429)
# ===========================================================================

class _FakeHandle:
    """Minimal seekable file-like to verify re-seek on retry."""
    def __init__(self):
        self.seek_calls = []
    def seek(self, pos):
        self.seek_calls.append(pos)


def _resp(status, json_body=None):
    r = MagicMock()
    r.status_code = status
    r.headers = {}
    r.json.return_value = json_body if json_body is not None else {"success": True, "result": {"id": "x"}}
    return r


def test_ckan_max_retries_default_and_env(monkeypatch):
    monkeypatch.delenv("CKAN_MAX_RETRIES", raising=False)
    assert utils._ckan_max_retries() == 5
    monkeypatch.setenv("CKAN_MAX_RETRIES", "2")
    assert utils._ckan_max_retries() == 2


def test_ckan_retries_502_then_succeeds(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "0")
    seq = [_resp(502), _resp(502), _resp(200)]
    with patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.requests, "post", side_effect=seq) as mock_post:
        result = utils.ckan_action_post("https://ckan.example", "resource_create", {"a": 1}, "Bearer t")
    assert result == {"id": "x"}
    assert mock_post.call_count == 3
    assert mock_sleep.call_count >= 2  # two backoff sleeps


def test_ckan_reseeks_file_handle_on_retry(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "0")
    handle = _FakeHandle()
    seq = [_resp(502), _resp(200)]
    with patch.object(utils.time, "sleep"), \
         patch.object(utils.requests, "post", side_effect=seq):
        utils.ckan_action_post("https://ckan.example", "resource_create", {"a": 1}, "Bearer t",
                               files={"upload": handle})
    # Sought to 0 before each of the 2 attempts.
    assert handle.seek_calls == [0, 0]


def test_ckan_exhausts_retries_raises(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "0")
    monkeypatch.setenv("CKAN_MAX_RETRIES", "2")
    with patch.object(utils.time, "sleep"), \
         patch.object(utils.requests, "post", return_value=_resp(502)):
        try:
            utils.ckan_action_post("https://ckan.example", "resource_create", {"a": 1}, None)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "502" in str(e)


def test_ckan_non_retryable_400_raises_immediately(monkeypatch):
    monkeypatch.setenv("CKAN_CALL_DELAY_SECONDS", "0")
    with patch.object(utils.time, "sleep") as mock_sleep, \
         patch.object(utils.requests, "post", return_value=_resp(400, {"success": False, "error": "bad"})):
        try:
            utils.ckan_action_post("https://ckan.example", "package_create", {"a": 1}, None)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
    mock_sleep.assert_not_called()
