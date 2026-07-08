"""Tests for tapis_links.py — Tapis postit minting and CKAN link resource building.

All Tapis and CKAN network calls are fully mocked.  No live services are used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure src/ is on the path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gam_registration import tapis_links  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_postit_response(redeem_url: str | None = None, postit_id: str | None = None):
    """Return a mock requests.Response for a successful postit creation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    result: dict = {}
    if redeem_url is not None:
        result["redeemUrl"] = redeem_url
    if postit_id is not None:
        result["id"] = postit_id
    mock_resp.json.return_value = {"result": result}
    return mock_resp


def _make_resource_plan(tmp_path: Path, filenames: list[str]) -> list[dict]:
    items = []
    for name in filenames:
        p = tmp_path / name
        p.write_text("dummy")
        suffix = Path(name).suffix.lower().lstrip(".")
        items.append({
            "resource_name": name,
            "resource_title": name,
            "resource_description": f"Description for {name}",
            "format": suffix.upper() if suffix else "BIN",
            "local_path": p,
            "relative_path": name,
        })
    return items


# ===========================================================================
# local_to_tapis_path
# ===========================================================================

class TestLocalToTapisPath:
    def test_returns_relative_posix_path(self, tmp_path):
        root = str(tmp_path)
        sub = tmp_path / "ygjk" / "Model_File" / "ygjk.nam"
        sub.parent.mkdir(parents=True)
        sub.touch()

        result = tapis_links.local_to_tapis_path(sub, root)
        assert result == "ygjk/Model_File/ygjk.nam"

    def test_no_leading_slash(self, tmp_path):
        root = str(tmp_path)
        f = tmp_path / "file.txt"
        f.touch()
        result = tapis_links.local_to_tapis_path(f, root)
        assert not result.startswith("/")

    def test_file_directly_in_root(self, tmp_path):
        root = str(tmp_path)
        f = tmp_path / "test.nam"
        f.touch()
        result = tapis_links.local_to_tapis_path(f, root)
        assert result == "test.nam"

    def test_raises_when_outside_root(self, tmp_path):
        root = tmp_path / "subdir"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.touch()

        with pytest.raises(ValueError, match="not under system_root_dir"):
            tapis_links.local_to_tapis_path(outside, str(root))

    def test_accepts_path_objects(self, tmp_path):
        root = tmp_path
        f = tmp_path / "a" / "b.txt"
        f.parent.mkdir()
        f.touch()
        result = tapis_links.local_to_tapis_path(f, root)
        assert result == "a/b.txt"

    def test_accepts_string_local_path(self, tmp_path):
        root = str(tmp_path)
        f = tmp_path / "f.dis"
        f.touch()
        result = tapis_links.local_to_tapis_path(str(f), root)
        assert result == "f.dis"


# ===========================================================================
# mint_postit_url
# ===========================================================================

class TestMintPostitUrl:
    _BASE_PARAMS = dict(
        system_id="corral-gam",
        tapis_path="ygjk/model/ygjk.nam",
        base_url="https://portals.tapis.io",
        jwt="test-jwt",
    )

    def test_returns_redeem_url_when_present(self):
        mock_resp = _make_mock_postit_response(redeem_url="https://portals.tapis.io/v3/files/postits/redeem/abc123")

        with patch("gam_registration.tapis_links.requests.post", return_value=mock_resp) as mock_post:
            url = tapis_links.mint_postit_url(**self._BASE_PARAMS)

        assert url == "https://portals.tapis.io/v3/files/postits/redeem/abc123"
        mock_post.assert_called_once()

    def test_constructs_url_from_id_when_no_redeem_url(self):
        mock_resp = _make_mock_postit_response(postit_id="xyz789")

        with patch("gam_registration.tapis_links.requests.post", return_value=mock_resp):
            url = tapis_links.mint_postit_url(**self._BASE_PARAMS)

        assert url == "https://portals.tapis.io/v3/files/postits/redeem/xyz789"

    def test_raises_when_neither_redeem_url_nor_id(self):
        mock_resp = _make_mock_postit_response()  # empty result

        with patch("gam_registration.tapis_links.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="redeemUrl.*id"):
                tapis_links.mint_postit_url(**self._BASE_PARAMS)

    def test_posts_to_correct_url(self):
        mock_resp = _make_mock_postit_response(redeem_url="https://example.com/redeem/1")

        with patch("gam_registration.tapis_links.requests.post", return_value=mock_resp) as mock_post:
            tapis_links.mint_postit_url(
                system_id="my-system",
                tapis_path="folder/file.txt",
                base_url="https://portals.tapis.io",
                jwt="my-jwt",
            )

        call_args = mock_post.call_args
        called_url = call_args[0][0]
        assert called_url == "https://portals.tapis.io/v3/files/postits/my-system/folder/file.txt"

    def test_bearer_header_sent(self):
        mock_resp = _make_mock_postit_response(redeem_url="https://example.com/r/1")

        with patch("gam_registration.tapis_links.requests.post", return_value=mock_resp) as mock_post:
            tapis_links.mint_postit_url(**self._BASE_PARAMS)

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers") or {}
        assert headers.get("Authorization") == "Bearer test-jwt"

    def test_query_params_sent(self):
        mock_resp = _make_mock_postit_response(redeem_url="https://example.com/r/1")

        with patch("gam_registration.tapis_links.requests.post", return_value=mock_resp) as mock_post:
            tapis_links.mint_postit_url(
                **self._BASE_PARAMS,
                allowed_uses=5,
                valid_seconds=86400,
            )

        params = mock_post.call_args.kwargs.get("params") or mock_post.call_args[1].get("params") or {}
        assert params["allowedUses"] == 5
        assert params["validSeconds"] == 86400

    def test_raises_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch("gam_registration.tapis_links.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="403"):
                tapis_links.mint_postit_url(**self._BASE_PARAMS)

    def test_retries_on_502_then_succeeds(self):
        """A 502 on the first call should be retried; the second call succeeds."""
        bad_resp = MagicMock()
        bad_resp.status_code = 502

        good_resp = _make_mock_postit_response(redeem_url="https://example.com/redeem/ok")

        with patch("gam_registration.tapis_links.requests.post", side_effect=[bad_resp, good_resp]) as mock_post, \
             patch("gam_registration.tapis_links.time.sleep"):
            url = tapis_links.mint_postit_url(**self._BASE_PARAMS)

        assert url == "https://example.com/redeem/ok"
        assert mock_post.call_count == 2

    def test_raises_after_exhausting_retries(self):
        """All retries returning 502 should eventually raise."""
        bad_resp = MagicMock()
        bad_resp.status_code = 502
        bad_resp.text = "Bad Gateway"

        # 6 attempts (1 initial + 5 retries).
        with patch("gam_registration.tapis_links.requests.post", return_value=bad_resp), \
             patch("gam_registration.tapis_links.time.sleep"):
            with pytest.raises(RuntimeError):
                tapis_links.mint_postit_url(**self._BASE_PARAMS)

    def test_429_triggers_retry(self):
        """A 429 response should also be retried."""
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429

        good_resp = _make_mock_postit_response(redeem_url="https://example.com/redeem/ok")

        with patch("gam_registration.tapis_links.requests.post", side_effect=[rate_limit_resp, good_resp]) as mock_post, \
             patch("gam_registration.tapis_links.time.sleep"):
            url = tapis_links.mint_postit_url(**self._BASE_PARAMS)

        assert url == "https://example.com/redeem/ok"
        assert mock_post.call_count == 2


# ===========================================================================
# build_tapis_link_resources
# ===========================================================================

class TestBuildTapisLinkResources:
    _BASE_KWARGS = dict(
        system_id="corral-gam",
        base_url="https://portals.tapis.io",
        jwt="test-jwt",
    )

    def test_returns_link_resource_dicts(self, tmp_path):
        plan = _make_resource_plan(tmp_path, ["file.nam", "file.dis"])

        def fake_mint(system_id, tapis_path, **kwargs):
            return f"https://portals.tapis.io/v3/files/postits/redeem/{tapis_path.replace('/', '_')}"

        with patch("gam_registration.tapis_links.mint_postit_url", side_effect=fake_mint), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "file.nam" in names
        assert "file.dis" in names

    def test_url_is_postit_url(self, tmp_path):
        plan = _make_resource_plan(tmp_path, ["ygjk.nam"])
        expected_url = "https://portals.tapis.io/v3/files/postits/redeem/abc"

        with patch("gam_registration.tapis_links.mint_postit_url", return_value=expected_url), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert results[0]["url"] == expected_url

    def test_name_preserved_from_resource_plan(self, tmp_path):
        plan = _make_resource_plan(tmp_path, ["ygjk.nam"])
        plan[0]["resource_name"] = "Custom / name"

        with patch("gam_registration.tapis_links.mint_postit_url", return_value="https://example.com/r/1"), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert results[0]["name"] == "Custom / name"
        assert results[0]["resource_name"] == "Custom / name"

    def test_mint_prefix_keys_passed_through(self, tmp_path):
        plan = _make_resource_plan(tmp_path, ["ygjk.nam"])
        plan[0]["mint_standard_variables"] = "groundwater__hydraulic_head"

        with patch("gam_registration.tapis_links.mint_postit_url", return_value="https://example.com/r/1"), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert results[0].get("mint_standard_variables") == "groundwater__hydraulic_head"

    def test_per_file_mint_failure_skips_file(self, tmp_path):
        """A mint failure on one file should skip it; others succeed."""
        plan = _make_resource_plan(tmp_path, ["good.nam", "bad.dis", "good2.bas"])

        def selective_mint(system_id, tapis_path, **kwargs):
            if "bad" in tapis_path:
                raise RuntimeError("Tapis error for bad file")
            return f"https://example.com/r/{tapis_path}"

        with patch("gam_registration.tapis_links.mint_postit_url", side_effect=selective_mint), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        # Only 2 of 3 files succeed.
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "bad.dis" not in names
        assert "good.nam" in names
        assert "good2.bas" in names

    def test_all_failures_returns_empty_list(self, tmp_path):
        plan = _make_resource_plan(tmp_path, ["a.nam", "b.dis"])

        with patch("gam_registration.tapis_links.mint_postit_url", side_effect=RuntimeError("fail")), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert results == []

    def test_format_derived_from_suffix(self, tmp_path):
        plan = _make_resource_plan(tmp_path, ["data.hds"])
        plan[0]["format"] = ""  # Clear format so it gets derived.

        with patch("gam_registration.tapis_links.mint_postit_url", return_value="https://example.com/r/1"), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert results[0]["format"] == "HDS"

    def test_format_preserved_when_already_set(self, tmp_path):
        plan = _make_resource_plan(tmp_path, ["data.hds"])
        plan[0]["format"] = "BINARY-OUTPUT"

        with patch("gam_registration.tapis_links.mint_postit_url", return_value="https://example.com/r/1"), \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                plan,
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert results[0]["format"] == "BINARY-OUTPUT"

    def test_empty_plan_returns_empty_list(self, tmp_path):
        with patch("gam_registration.tapis_links.mint_postit_url") as mock_mint, \
             patch("gam_registration.tapis_links.time.sleep"):
            results = tapis_links.build_tapis_link_resources(
                [],
                system_root_dir=str(tmp_path),
                **self._BASE_KWARGS,
            )

        assert results == []
        mock_mint.assert_not_called()


# ===========================================================================
# refresh_postit_urls
# ===========================================================================

class TestRefreshPostitUrls:
    def test_returns_fresh_urls(self):
        pairs = [("sys", "path/a.nam"), ("sys", "path/b.dis")]

        def fake_mint(system_id, tapis_path, **kwargs):
            return f"https://example.com/fresh/{tapis_path}"

        with patch("gam_registration.tapis_links.mint_postit_url", side_effect=fake_mint), \
             patch("gam_registration.tapis_links.time.sleep"):
            urls = tapis_links.refresh_postit_urls(
                pairs,
                base_url="https://portals.tapis.io",
                jwt="fresh-jwt",
            )

        assert len(urls) == 2
        assert urls[0] == "https://example.com/fresh/path/a.nam"
        assert urls[1] == "https://example.com/fresh/path/b.dis"

    def test_returns_none_on_failure(self):
        pairs = [("sys", "good.nam"), ("sys", "bad.dis")]

        def selective_mint(system_id, tapis_path, **kwargs):
            if "bad" in tapis_path:
                raise RuntimeError("fail")
            return "https://example.com/ok"

        with patch("gam_registration.tapis_links.mint_postit_url", side_effect=selective_mint), \
             patch("gam_registration.tapis_links.time.sleep"):
            urls = tapis_links.refresh_postit_urls(
                pairs,
                base_url="https://portals.tapis.io",
                jwt="jwt",
            )

        assert urls[0] == "https://example.com/ok"
        assert urls[1] is None


# ===========================================================================
# orchestrate wiring: register_by_reference=True vs False
# ===========================================================================

class TestOrchestrateRegisterByReference:
    """When register_by_reference=True + approval=REGISTER, create_link_resources
    is called with postit-url resources and upsert_resources is NOT called.
    When register_by_reference=False, the byte-upload path is unchanged.
    """

    def _run_with_ref(
        self,
        tmp_path: Path,
        register_by_reference: bool,
        monkeypatch,
        *,
        tapis_system_id: str = "corral-gam",
        tapis_rootdir: str | None = None,
    ):
        """Helper: run orchestrate.run_registration in REGISTER mode with full mocks."""
        import gam_registration.orchestrate as orchestrate
        from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound

        if tapis_rootdir is None:
            tapis_rootdir = str(tmp_path)

        # Set up env vars needed for by-reference mode.
        monkeypatch.setenv("TAPIS_SYSTEM_ID", tapis_system_id)
        monkeypatch.setenv("TAPIS_SYSTEM_ROOTDIR", tapis_rootdir)
        monkeypatch.setenv("TAPIS_FILES_BASE_URL", "https://portals.tapis.io")
        monkeypatch.setenv("POSTIT_VALID_SECONDS", "3153600000")
        monkeypatch.setenv("POSTIT_ALLOWED_USES", "-1")

        # Create a minimal model file.
        f = tmp_path / "test.nam"
        f.write_text("# test\n")

        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
            "title": "Test GAM",
            "twdb_page_url": "https://twdb.texas.gov/test",
            "report_url": "",
            "boundary_bbox_geojson": "",
        }

        evaluator = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
        loop_round = LoopRound(
            round_number=1,
            candidate_metadata={"dataset_title": "Test GAM"},
            fair_evaluator=evaluator,
            usability_evaluator=evaluator,
            converged=True,
        )
        persona_result = PersonaLoopResult(
            converged=True,
            rounds=1,
            proposed_metadata={"dataset_title": "Test GAM"},
            transcript=[loop_round],
            stop_reason="converged",
            model_id="test-gam",
            timestamp="20260625_120000",
        )

        mock_dataset = {"id": "pkg-abc", "name": "test-gam", "resources": []}
        postit_url = "https://portals.tapis.io/v3/files/postits/redeem/xyz"

        resource_plan = [
            {
                "resource_name": "test.nam",
                "resource_title": "test.nam",
                "resource_description": "MODFLOW namefile",
                "format": "NAM",
                "local_path": f,
                "relative_path": "test.nam",
                "source_url": "",
            }
        ]

        captured: dict = {}

        def fake_create_link_resources(ckan_url, dataset, link_resources, auth_header):
            captured["link_resources"] = link_resources
            return [{"id": "r1", "name": "test.nam"}]

        def fake_upsert_resources(ckan_url, dataset, rp, auth_header, **kwargs):
            captured["upsert_called"] = True
            return [], 0, 0

        def fake_build_tapis_links(rp, *, system_id, system_root_dir, base_url, jwt, **kwargs):
            return [
                {
                    "resource_name": item["resource_name"],
                    "name": item["resource_name"],
                    "url": postit_url,
                    "description": "desc",
                    "format": item.get("format", "BIN"),
                }
                for item in rp
            ]

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[f]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=resource_plan), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={"dataset_name": "test-gam"}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam", "extras": []}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate._u.create_or_update_ckan_dataset", return_value=mock_dataset), \
             patch("gam_registration.orchestrate._u.create_link_resources", side_effect=fake_create_link_resources), \
             patch("gam_registration.orchestrate._u.upsert_resources", side_effect=fake_upsert_resources), \
             patch("gam_registration.orchestrate._tl.build_tapis_link_resources", side_effect=fake_build_tapis_links), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                auth_header="Bearer test-jwt-token",
                llm_model="m",
                llm_api_key="k",
                approval="REGISTER",
                register_by_reference=register_by_reference,
            )

        return result, captured

    def test_by_reference_true_calls_create_link_resources_not_upsert(self, tmp_path, monkeypatch):
        result, captured = self._run_with_ref(tmp_path, register_by_reference=True, monkeypatch=monkeypatch)

        assert result.ok is True, f"Expected ok=True, got error={result.error}"
        # upsert_resources must NOT have been called.
        assert "upsert_called" not in captured, "upsert_resources should NOT be called in by-reference mode"
        # create_link_resources must have been called.
        assert "link_resources" in captured, "create_link_resources should have been called"

    def test_by_reference_link_resources_have_postit_urls(self, tmp_path, monkeypatch):
        result, captured = self._run_with_ref(tmp_path, register_by_reference=True, monkeypatch=monkeypatch)

        assert result.ok is True
        link_resources = captured.get("link_resources", [])
        # At least one link resource should be a Tapis postit URL.
        postit_links = [lr for lr in link_resources if "postits/redeem" in (lr.get("url") or "")]
        assert len(postit_links) >= 1, f"Expected postit URL in link_resources; got {link_resources}"

    def test_by_reference_false_uses_byte_upload(self, tmp_path, monkeypatch):
        """With register_by_reference=False, upsert_resources IS called."""
        result, captured = self._run_with_ref(tmp_path, register_by_reference=False, monkeypatch=monkeypatch)

        assert result.ok is True
        assert "upsert_called" in captured, "upsert_resources should be called in byte-upload mode"

    def test_by_reference_missing_system_id_raises(self, tmp_path, monkeypatch):
        """Missing TAPIS_SYSTEM_ID should raise RuntimeError (not silent fallback)."""
        import gam_registration.orchestrate as orchestrate
        from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound

        # Set ROOTDIR but NOT SYSTEM_ID.
        monkeypatch.delenv("TAPIS_SYSTEM_ID", raising=False)
        monkeypatch.setenv("TAPIS_SYSTEM_ROOTDIR", str(tmp_path))

        f = tmp_path / "test.nam"
        f.write_text("# test\n")
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
        }

        evaluator = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
        loop_round = LoopRound(
            round_number=1,
            candidate_metadata={},
            fair_evaluator=evaluator,
            usability_evaluator=evaluator,
            converged=True,
        )
        persona_result = PersonaLoopResult(
            converged=True, rounds=1, proposed_metadata={}, transcript=[loop_round],
            stop_reason="converged", model_id="test-gam", timestamp="20260625_120000",
        )
        mock_dataset = {"id": "pkg-abc", "name": "test-gam", "resources": []}

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[f]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[{"resource_name": "test.nam", "local_path": f, "format": "NAM", "source_url": ""}]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={"dataset_name": "test-gam"}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam", "extras": []}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate._u.create_or_update_ckan_dataset", return_value=mock_dataset), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                auth_header="Bearer valid-jwt",
                llm_model="m",
                llm_api_key="k",
                approval="REGISTER",
                register_by_reference=True,
            )

        # Should fail with a clear error about missing env var.
        assert result.ok is False
        assert "TAPIS_SYSTEM_ID" in result.error or "TAPIS_SYSTEM_ROOTDIR" in result.error

    def test_by_reference_non_bearer_auth_raises(self, tmp_path, monkeypatch):
        """Non-Bearer auth_header should raise a clear RuntimeError."""
        import gam_registration.orchestrate as orchestrate
        from gam_registration.persona_loop import PersonaLoopResult, EvaluatorVerdict, LoopRound

        monkeypatch.setenv("TAPIS_SYSTEM_ID", "corral-gam")
        monkeypatch.setenv("TAPIS_SYSTEM_ROOTDIR", str(tmp_path))

        f = tmp_path / "test.nam"
        f.write_text("# test\n")
        model_record = {
            "package_id": "test-gam",
            "package_folder": str(tmp_path),
        }

        evaluator = EvaluatorVerdict(verdict="pass", questions=[], recommendations=[])
        loop_round = LoopRound(
            round_number=1,
            candidate_metadata={},
            fair_evaluator=evaluator,
            usability_evaluator=evaluator,
            converged=True,
        )
        persona_result = PersonaLoopResult(
            converged=True, rounds=1, proposed_metadata={}, transcript=[loop_round],
            stop_reason="converged", model_id="test-gam", timestamp="20260625_120000",
        )
        mock_dataset = {"id": "pkg-abc", "name": "test-gam", "resources": []}

        with patch("gam_registration.orchestrate._u.list_resource_files", return_value=[f]), \
             patch("gam_registration.orchestrate._u.build_resource_plan", return_value=[{"resource_name": "test.nam", "local_path": f, "format": "NAM", "source_url": ""}]), \
             patch("gam_registration.orchestrate._u.annotate_resource_plan_with_mint_standard_variables", side_effect=lambda rp, **kw: rp), \
             patch("gam_registration.orchestrate._u.fetch_source_metadata", return_value={"excerpt": "", "url": "", "title": "", "meta_description": ""}), \
             patch("gam_registration.orchestrate._twdb.discover_report_url", return_value=None), \
             patch("gam_registration.orchestrate._u.propose_ckan_dataset_metadata_with_llm", return_value={"dataset_name": "test-gam"}), \
             patch("gam_registration.orchestrate._pl.run_persona_metadata_loop", return_value=persona_result), \
             patch("gam_registration.orchestrate._sm.map_to_subside_dataset", return_value={"type": "subside_dataset", "name": "test-gam", "extras": []}), \
             patch("gam_registration.orchestrate._twdb.build_link_resources", return_value=[]), \
             patch("gam_registration.orchestrate._u.create_or_update_ckan_dataset", return_value=mock_dataset), \
             patch("gam_registration.orchestrate.time.sleep"):

            result = orchestrate.run_registration(
                model_record,
                ckan_url="https://ckan.example.com",
                auth_header="plain-api-key",  # NOT a Bearer token
                llm_model="m",
                llm_api_key="k",
                approval="REGISTER",
                register_by_reference=True,
            )

        assert result.ok is False
        assert "Bearer" in result.error or "tapis_password" in result.error

    def test_apply_result_includes_register_by_reference_flag(self, tmp_path, monkeypatch):
        result, _ = self._run_with_ref(tmp_path, register_by_reference=True, monkeypatch=monkeypatch)

        assert result.ok is True
        assert result.apply_result is not None
        assert result.apply_result.get("register_by_reference") is True
        assert "postits_minted" in result.apply_result
