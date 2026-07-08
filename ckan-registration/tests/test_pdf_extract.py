"""Tests for pdf_extract.py — map-reduce PDF pipeline.

All network calls, LLM calls, and pymupdf (fitz) usage are mocked.
Tests cover:
  - fetch_pdf_to_temp: content-type validation, size cap, happy path
  - extract_pdf_text_chunks: chunk splitting, max_chunks cap + logging
  - extract_metadata_from_chunk: MAP step — LLM call shape, output structure
  - consolidate_chunk_metadata: REDUCE step — LLM call shape, output structure
  - run_pdf_map_reduce: orchestration order, inter-call sleep
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path so gam_registration package is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Module under test — import AFTER patching utils if needed.
# ---------------------------------------------------------------------------
import gam_registration.pdf_extract as pdf_extract  # noqa: E402


# ===========================================================================
# fetch_pdf_to_temp
# ===========================================================================

class TestFetchPdfToTemp:
    """Tests for fetch_pdf_to_temp."""

    def _make_mock_response(
        self,
        status_code: int = 200,
        content_type: str = "application/pdf",
        content: bytes = b"%PDF-1.4 fake-pdf-content",
        headers: dict | None = None,
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = {"Content-Type": content_type, **(headers or {})}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        # iter_content yields the content in one shot.
        resp.iter_content = MagicMock(return_value=iter([content]))
        return resp

    def test_happy_path_returns_existing_file(self, tmp_path):
        """fetch_pdf_to_temp returns a Path that exists and contains PDF bytes."""
        mock_resp = self._make_mock_response()
        with patch("gam_registration.pdf_extract.requests.get", return_value=mock_resp) as mock_get:
            result = pdf_extract.fetch_pdf_to_temp("https://example.com/report.pdf")
        try:
            assert isinstance(result, Path)
            assert result.exists()
            data = result.read_bytes()
            assert data == b"%PDF-1.4 fake-pdf-content"
        finally:
            result.unlink(missing_ok=True)

    def test_rejects_non_pdf_content_type(self):
        """fetch_pdf_to_temp raises ValueError for non-PDF content type."""
        mock_resp = self._make_mock_response(content_type="text/html")
        with patch("gam_registration.pdf_extract.requests.get", return_value=mock_resp):
            with pytest.raises(ValueError, match="content-type"):
                pdf_extract.fetch_pdf_to_temp("https://example.com/page.html")

    def test_allows_octet_stream_content_type(self):
        """fetch_pdf_to_temp accepts application/octet-stream (servers may omit PDF type)."""
        mock_resp = self._make_mock_response(content_type="application/octet-stream")
        with patch("gam_registration.pdf_extract.requests.get", return_value=mock_resp):
            result = pdf_extract.fetch_pdf_to_temp("https://example.com/report.pdf")
        try:
            assert result.exists()
        finally:
            result.unlink(missing_ok=True)

    def test_raises_on_size_cap_exceeded(self):
        """fetch_pdf_to_temp raises ValueError when the download exceeds the size cap."""
        # Patch the cap to a tiny value so the test doesn't need to stream 200 MB.
        big_chunk = b"x" * 1024
        mock_resp = self._make_mock_response(content=big_chunk)
        # Make iter_content return many chunks.
        mock_resp.iter_content = MagicMock(
            return_value=iter([big_chunk] * 300)
        )
        original_cap = pdf_extract._MAX_PDF_BYTES
        pdf_extract._MAX_PDF_BYTES = 10  # 10 bytes cap for test.
        try:
            with patch("gam_registration.pdf_extract.requests.get", return_value=mock_resp):
                with pytest.raises(ValueError, match="exceeded"):
                    pdf_extract.fetch_pdf_to_temp("https://example.com/big.pdf")
        finally:
            pdf_extract._MAX_PDF_BYTES = original_cap

    def test_http_error_propagates(self):
        """fetch_pdf_to_temp propagates HTTP errors from raise_for_status."""
        mock_resp = self._make_mock_response(status_code=404)
        with patch("gam_registration.pdf_extract.requests.get", return_value=mock_resp):
            with pytest.raises(Exception):
                pdf_extract.fetch_pdf_to_temp("https://example.com/missing.pdf")


# ===========================================================================
# extract_pdf_text_chunks
# ===========================================================================

class TestExtractPdfTextChunks:
    """Tests for extract_pdf_text_chunks."""

    def _make_mock_fitz(self, page_texts: list[str]):
        """Build a fake fitz module with pages that return the given texts."""
        mock_page = MagicMock()

        def _get_text_side_effect():
            # pop from front of list each call.
            return page_texts.pop(0) if page_texts else ""

        mock_page.get_text = MagicMock(side_effect=_get_text_side_effect)

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(return_value=mock_doc)
        return mock_fitz

    def test_basic_chunking(self, tmp_path):
        """Returns chunks; each chunk is within chunk_chars."""
        page_texts = ["A" * 5000, "B" * 5000, "C" * 5000]
        mock_fitz = self._make_mock_fitz(page_texts.copy())
        with patch.object(pdf_extract, "fitz", mock_fitz), \
             patch.object(pdf_extract, "_lazy_fitz", return_value=mock_fitz):
            chunks = pdf_extract.extract_pdf_text_chunks(
                tmp_path / "dummy.pdf", chunk_chars=8500
            )
        assert len(chunks) > 0
        for chunk in chunks:
            # Allow a small buffer for whitespace splitting.
            assert len(chunk) <= 8500 + 10

    def test_max_chunks_cap_applied_and_logged(self, tmp_path, caplog):
        """When max_chunks is exceeded, only first max_chunks returned and a warning is logged."""
        # 3 pages * 5000 chars = 15000 chars → ~2 chunks at 8500 each.
        page_texts_factory = lambda: ["A " * 2500, "B " * 2500, "C " * 2500]
        mock_fitz = self._make_mock_fitz(page_texts_factory())

        with patch.object(pdf_extract, "fitz", mock_fitz), \
             patch.object(pdf_extract, "_lazy_fitz", return_value=mock_fitz), \
             caplog.at_level(logging.WARNING, logger="pdf_extract"):
            chunks = pdf_extract.extract_pdf_text_chunks(
                tmp_path / "dummy.pdf", chunk_chars=8500, max_chunks=1
            )

        assert len(chunks) == 1
        # Confirm the cap warning was logged.
        assert any("max_chunks" in r.message for r in caplog.records), (
            f"Expected 'max_chunks' warning in log, got: {[r.message for r in caplog.records]}"
        )

    def test_empty_pdf_returns_empty_list(self, tmp_path, caplog):
        """Returns empty list when no text is extractable."""
        mock_fitz = self._make_mock_fitz(["", "", ""])
        with patch.object(pdf_extract, "fitz", mock_fitz), \
             patch.object(pdf_extract, "_lazy_fitz", return_value=mock_fitz), \
             caplog.at_level(logging.WARNING, logger="pdf_extract"):
            result = pdf_extract.extract_pdf_text_chunks(tmp_path / "empty.pdf")
        assert result == []

    def test_fitz_open_failure_returns_empty_list(self, tmp_path, caplog):
        """Returns empty list when pymupdf fails to open the file."""
        mock_fitz = MagicMock()
        mock_fitz.open = MagicMock(side_effect=RuntimeError("cannot open"))
        with patch.object(pdf_extract, "fitz", mock_fitz), \
             patch.object(pdf_extract, "_lazy_fitz", return_value=mock_fitz), \
             caplog.at_level(logging.WARNING, logger="pdf_extract"):
            result = pdf_extract.extract_pdf_text_chunks(tmp_path / "broken.pdf")
        assert result == []


# ===========================================================================
# extract_metadata_from_chunk (MAP step)
# ===========================================================================

class TestExtractMetadataFromChunk:
    """Tests for extract_metadata_from_chunk."""

    _CHUNK = "The Blossom Aquifer Groundwater Availability Model was developed in 2005."
    _FIELDS = ["dataset_title", "temporal_coverage_start", "author"]

    def test_sends_chunk_and_fields_to_llm(self):
        """extract_metadata_from_chunk passes chunk text and schema_hint to the LLM helper."""
        fake_response = {
            "dataset_title": "Blossom Aquifer GAM",
            "temporal_coverage_start": "2005",
            "author": None,
            "chunk_summary": "GAM development year.",
        }
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value='{"dataset_title": "Blossom Aquifer GAM", "temporal_coverage_start": "2005", "author": null, "chunk_summary": "GAM development year."}') as mock_llm:
            result = pdf_extract.extract_metadata_from_chunk(
                self._CHUNK,
                self._FIELDS,
                llm_model="test-model",
                llm_api_key="test-key",
                llm_base_url="https://ai.example.com",
            )
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args.kwargs
        # Verify fields are in user_payload.
        user_payload = call_kwargs["user_payload"]
        assert "fields_to_extract" in user_payload
        assert self._FIELDS == user_payload["fields_to_extract"] or \
               set(self._FIELDS).issubset(set(user_payload.get("fields_to_extract", [])))
        assert self._CHUNK in str(user_payload)

    def test_returns_dict_with_all_requested_fields(self):
        """Result contains all requested fields plus chunk_summary."""
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value='{"dataset_title": "GAM Report", "temporal_coverage_start": null, "author": null, "chunk_summary": "Intro."}'):
            result = pdf_extract.extract_metadata_from_chunk(
                self._CHUNK,
                self._FIELDS,
                llm_model="m",
                llm_api_key="k",
            )
        for field in self._FIELDS:
            assert field in result, f"Field {field!r} missing from result"
        assert "chunk_summary" in result

    def test_missing_fields_set_to_none(self):
        """Fields absent from LLM response are set to None (not KeyError)."""
        # LLM returns only partial data.
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value='{"dataset_title": "Test"}'):
            result = pdf_extract.extract_metadata_from_chunk(
                self._CHUNK,
                self._FIELDS,
                llm_model="m",
                llm_api_key="k",
            )
        assert result["temporal_coverage_start"] is None
        assert result["author"] is None

    def test_passes_model_and_api_key(self):
        """LLM helper receives the correct model and api_key."""
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value="{}") as mock_llm:
            pdf_extract.extract_metadata_from_chunk(
                "text",
                ["title"],
                llm_model="llama-3",
                llm_api_key="secret-key",
                llm_base_url="https://ai.example.com",
            )
        kwargs = mock_llm.call_args.kwargs
        assert kwargs["model"] == "llama-3"
        assert kwargs["api_key"] == "secret-key"
        assert kwargs["base_url"] == "https://ai.example.com"


# ===========================================================================
# consolidate_chunk_metadata (REDUCE step)
# ===========================================================================

class TestConsolidateChunkMetadata:
    """Tests for consolidate_chunk_metadata."""

    _CHUNK_RESULTS = [
        {
            "chunk_summary": "Introduction to Blossom Aquifer.",
            "dataset_title": "Blossom Aquifer GAM",
            "temporal_coverage_start": None,
        },
        {
            "chunk_summary": "Model calibration period: 1980-2005.",
            "dataset_title": None,
            "temporal_coverage_start": "1980",
        },
    ]

    def test_passes_chunk_results_to_llm(self):
        """consolidate_chunk_metadata passes all chunk results to the LLM."""
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value='{"dataset_title": "Blossom Aquifer GAM", "temporal_coverage_start": "1980"}') as mock_llm:
            result = pdf_extract.consolidate_chunk_metadata(
                self._CHUNK_RESULTS,
                landing_page_excerpt="TWDB Blossom page.",
                llm_model="m",
                llm_api_key="k",
            )
        mock_llm.assert_called_once()
        user_payload = mock_llm.call_args.kwargs["user_payload"]
        assert "per_chunk_candidates" in user_payload
        assert len(user_payload["per_chunk_candidates"]) == 2

    def test_passes_landing_page_excerpt(self):
        """Landing page excerpt is included in the LLM user payload."""
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value="{}") as mock_llm:
            pdf_extract.consolidate_chunk_metadata(
                self._CHUNK_RESULTS,
                landing_page_excerpt="<excerpt>TWDB page content</excerpt>",
                llm_model="m",
                llm_api_key="k",
            )
        user_payload = mock_llm.call_args.kwargs["user_payload"]
        assert "TWDB page content" in str(user_payload)

    def test_returns_parsed_dict(self):
        """Returns a dict parsed from the LLM JSON response."""
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value='{"dataset_title": "Blossom", "temporal_coverage_start": "1980"}'):
            result = pdf_extract.consolidate_chunk_metadata(
                self._CHUNK_RESULTS,
                llm_model="m",
                llm_api_key="k",
            )
        assert isinstance(result, dict)
        assert result.get("dataset_title") == "Blossom"
        assert result.get("temporal_coverage_start") == "1980"

    def test_chunk_summaries_included_in_payload(self):
        """chunk_summary strings from each chunk result are passed to the REDUCE call."""
        with patch("gam_registration.pdf_extract._chat_completion_content", return_value="{}") as mock_llm:
            pdf_extract.consolidate_chunk_metadata(
                self._CHUNK_RESULTS,
                llm_model="m",
                llm_api_key="k",
            )
        payload_str = str(mock_llm.call_args.kwargs["user_payload"])
        assert "Introduction to Blossom" in payload_str
        assert "1980-2005" in payload_str


# ===========================================================================
# run_pdf_map_reduce — orchestration order and sleep
# ===========================================================================

class TestRunPdfMapReduce:
    """Tests for run_pdf_map_reduce orchestration."""

    def test_map_then_reduce_called_in_order(self, tmp_path):
        """MAP calls precede the REDUCE call; order of LLM calls is correct."""
        dummy_pdf = tmp_path / "report.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        chunks = ["chunk one", "chunk two"]
        map_result = {"chunk_summary": "summary", "dataset_title": "Test GAM"}
        reduce_result = {"dataset_title": "Test GAM Final", "temporal_coverage_start": "2000"}

        call_log: list[str] = []

        def mock_extract_chunks(path, **kwargs):
            call_log.append("extract_chunks")
            return chunks

        def mock_map(chunk, schema, **kwargs):
            call_log.append(f"map:{chunk[:5]}")
            return map_result

        def mock_reduce(results, landing_page_excerpt="", **kwargs):
            call_log.append("reduce")
            return reduce_result

        with patch("gam_registration.pdf_extract.extract_pdf_text_chunks", side_effect=mock_extract_chunks), \
             patch("gam_registration.pdf_extract.extract_metadata_from_chunk", side_effect=mock_map), \
             patch("gam_registration.pdf_extract.consolidate_chunk_metadata", side_effect=mock_reduce), \
             patch("gam_registration.pdf_extract.time.sleep"):  # suppress actual sleep
            result = pdf_extract.run_pdf_map_reduce(
                dummy_pdf,
                "landing excerpt",
                llm_model="m",
                llm_api_key="k",
            )

        assert call_log[0] == "extract_chunks"
        # Both MAP calls come before REDUCE.
        map_calls = [e for e in call_log if e.startswith("map:")]
        assert len(map_calls) == 2
        reduce_idx = call_log.index("reduce")
        for i, entry in enumerate(call_log):
            if entry.startswith("map:"):
                assert i < reduce_idx, "MAP call appeared after REDUCE"

        assert result == reduce_result

    def test_inter_call_sleep_called(self, tmp_path):
        """time.sleep is called between LLM calls."""
        dummy_pdf = tmp_path / "report.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        with patch("gam_registration.pdf_extract.extract_pdf_text_chunks", return_value=["c1", "c2", "c3"]), \
             patch("gam_registration.pdf_extract.extract_metadata_from_chunk", return_value={"chunk_summary": "s"}), \
             patch("gam_registration.pdf_extract.consolidate_chunk_metadata", return_value={}), \
             patch("gam_registration.pdf_extract.time.sleep") as mock_sleep:
            pdf_extract.run_pdf_map_reduce(dummy_pdf, llm_model="m", llm_api_key="k")

        # sleep should be called at least once (between MAP calls and before REDUCE).
        assert mock_sleep.call_count >= 1

    def test_returns_empty_dict_when_no_chunks(self, tmp_path):
        """Returns empty dict and skips LLM calls when no text could be extracted."""
        dummy_pdf = tmp_path / "report.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        with patch("gam_registration.pdf_extract.extract_pdf_text_chunks", return_value=[]), \
             patch("gam_registration.pdf_extract.extract_metadata_from_chunk") as mock_map, \
             patch("gam_registration.pdf_extract.consolidate_chunk_metadata") as mock_reduce:
            result = pdf_extract.run_pdf_map_reduce(dummy_pdf, llm_model="m", llm_api_key="k")

        assert result == {}
        mock_map.assert_not_called()
        mock_reduce.assert_not_called()

    def test_max_chunks_forwarded_to_extract_chunks(self, tmp_path):
        """max_chunks parameter is forwarded to extract_pdf_text_chunks."""
        dummy_pdf = tmp_path / "report.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        with patch("gam_registration.pdf_extract.extract_pdf_text_chunks", return_value=[]) as mock_extract, \
             patch("gam_registration.pdf_extract.consolidate_chunk_metadata", return_value={}), \
             patch("gam_registration.pdf_extract.time.sleep"):
            pdf_extract.run_pdf_map_reduce(
                dummy_pdf, llm_model="m", llm_api_key="k", max_chunks=5
            )

        kwargs = mock_extract.call_args.kwargs
        assert kwargs.get("max_chunks") == 5


# ===========================================================================
# Lazy import: module importable without pymupdf
# ===========================================================================

def test_module_importable_without_pymupdf():
    """pdf_extract imports without error even when pymupdf is absent."""
    import importlib
    import sys

    # Temporarily hide fitz from sys.modules to simulate absence.
    orig_fitz = sys.modules.pop("fitz", None)
    try:
        # Reload the module to re-exercise module-level code.
        importlib.reload(pdf_extract)
        # Module-level attributes should still be accessible.
        assert hasattr(pdf_extract, "fetch_pdf_to_temp")
        assert hasattr(pdf_extract, "extract_pdf_text_chunks")
    finally:
        if orig_fitz is not None:
            sys.modules["fitz"] = orig_fitz
        # Restore pdf_extract to a clean state.
        importlib.reload(pdf_extract)


# ===========================================================================
# find_local_report_pdf
# ===========================================================================

class TestFindLocalReportPdf:
    """Tests for find_local_report_pdf — ranked PDF finder (no network)."""

    def test_returns_none_when_no_pdf_exists(self, tmp_path):
        """Returns None when no PDF file is present anywhere under package_dir."""
        (tmp_path / "Model_File").mkdir()
        (tmp_path / "Model_File" / "model.nam").write_text("# namefile")
        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result is None

    def test_returns_none_for_missing_directory(self, tmp_path):
        """Returns None gracefully when package_dir does not exist."""
        result = pdf_extract.find_local_report_pdf(tmp_path / "nonexistent")
        assert result is None

    def test_returns_none_for_none_input(self):
        """Returns None when called with None."""
        result = pdf_extract.find_local_report_pdf(None)  # type: ignore[arg-type]
        assert result is None

    def test_tier_a_report_dir_preferred_over_generic_pdf(self, tmp_path):
        """A PDF inside a 'Report/' dir is chosen over one in another dir."""
        report_dir = tmp_path / "Report"
        report_dir.mkdir()
        report_pdf = report_dir / "gam_report.pdf"
        report_pdf.write_bytes(b"%PDF-1.4 report content here")

        other_dir = tmp_path / "Model_File"
        other_dir.mkdir()
        other_pdf = other_dir / "other.pdf"
        other_pdf.write_bytes(b"%PDF-1.4 " + b"x" * 5000)  # larger, but not in Report/

        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == report_pdf

    def test_tier_a_reports_dir_also_recognized(self, tmp_path):
        """A PDF inside a 'Reports/' directory (plural) is also tier-a."""
        reports_dir = tmp_path / "Reports"
        reports_dir.mkdir()
        pdf = reports_dir / "technical_report.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        other_dir = tmp_path / "Data"
        other_dir.mkdir()
        other = other_dir / "output.pdf"
        other.write_bytes(b"%PDF-1.4 " + b"x" * 9000)

        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == pdf

    def test_tier_a_case_insensitive_dir_name(self, tmp_path):
        """'REPORT/' and 'report/' directories both count as tier-a."""
        upper_report_dir = tmp_path / "REPORT"
        upper_report_dir.mkdir()
        pdf = upper_report_dir / "file.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        other_dir = tmp_path / "Grid"
        other_dir.mkdir()
        bigger = other_dir / "bigfile.pdf"
        bigger.write_bytes(b"%PDF-1.4 " + b"x" * 10000)

        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == pdf

    def test_tier_b_report_filename_preferred_over_generic(self, tmp_path):
        """When no 'Report/' dir exists, a PDF with 'report' in the name is preferred."""
        data_dir = tmp_path / "Data"
        data_dir.mkdir()

        generic_pdf = data_dir / "model_output.pdf"
        generic_pdf.write_bytes(b"%PDF-1.4 " + b"x" * 9000)  # larger

        report_pdf = data_dir / "gam_report_2023.pdf"
        report_pdf.write_bytes(b"%PDF-1.4 small")

        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == report_pdf

    def test_tier_c_largest_returned_when_no_report_naming(self, tmp_path):
        """When no 'Report/' dir and no 'report'-named PDFs, returns the largest PDF."""
        other_dir = tmp_path / "Other"
        other_dir.mkdir()

        small = other_dir / "alpha.pdf"
        small.write_bytes(b"%PDF-1.4 small content")

        large = other_dir / "zeta.pdf"
        large.write_bytes(b"%PDF-1.4 " + b"x" * 5000)

        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == large

    def test_nested_report_dir_found_recursively(self, tmp_path):
        """PDFs in a nested 'Report/' subdir are discovered recursively."""
        nested = tmp_path / "Yegua_GAM" / "Report"
        nested.mkdir(parents=True)
        pdf = nested / "technical.pdf"
        pdf.write_bytes(b"%PDF-1.4 report content")

        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == pdf

    def test_largest_in_tier_a_selected_when_multiple(self, tmp_path):
        """When multiple PDFs are in 'Report/', the largest is returned."""
        report_dir = tmp_path / "Report"
        report_dir.mkdir()

        small = report_dir / "appendix.pdf"
        small.write_bytes(b"%PDF-1.4 short")

        large = report_dir / "main_report.pdf"
        large.write_bytes(b"%PDF-1.4 " + b"x" * 5000)

        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == large

    def test_uppercase_pdf_extension_found(self, tmp_path):
        """PDFs with a .PDF extension (uppercase) are discovered."""
        report_dir = tmp_path / "Report"
        report_dir.mkdir()
        pdf = report_dir / "REPORT.PDF"
        pdf.write_bytes(b"%PDF-1.4")
        result = pdf_extract.find_local_report_pdf(tmp_path)
        assert result == pdf
