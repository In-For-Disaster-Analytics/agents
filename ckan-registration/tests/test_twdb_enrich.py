"""Tests for twdb_enrich.py — report discovery and link resources.

All network calls are mocked.
Tests cover:
  - discover_report_url: found / not-found / override-precedence / fetch-error
  - build_link_resources: resource shape, missing inputs, partial inputs
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on path so gam_registration package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gam_registration.twdb_enrich as twdb_enrich  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

# A minimal TWDB landing-page HTML with a PDF report link.
LANDING_PAGE_HTML_WITH_REPORT = """\
<html>
<head><title>Blossom Aquifer GAM</title></head>
<body>
<h1>Blossom Aquifer Groundwater Availability Model</h1>
<p>This page provides information about the Blossom Aquifer GAM.</p>
<ul>
  <li><a href="blsm_gam_report.pdf">Final Report</a></li>
  <li><a href="blsm_figures.pdf">Figures</a></li>
  <li><a href="index.htm">Back to GAM Home</a></li>
</ul>
</body>
</html>
"""

# HTML with no PDF links.
LANDING_PAGE_HTML_NO_PDF = """\
<html>
<body>
<h1>Aquifer Info</h1>
<p>No PDFs here.</p>
<a href="map.html">Map</a>
</body>
</html>
"""

# HTML with multiple PDF links; the one with "final report" text should win.
LANDING_PAGE_HTML_MULTIPLE_PDFS = """\
<html>
<body>
<a href="appendix_a.pdf">Appendix A</a>
<a href="exec_summary.pdf">Executive Summary</a>
<a href="final_report_full.pdf">Final Report</a>
<a href="maps.pdf">Maps</a>
</body>
</html>
"""

LANDING_PAGE_URL = "https://www.twdb.texas.gov/groundwater/models/gam/blsm/blsm.asp"


# ===========================================================================
# discover_report_url
# ===========================================================================

class TestDiscoverReportUrl:
    """Tests for discover_report_url."""

    # --- Override precedence ---

    def test_override_takes_precedence_over_discovery(self):
        """When report_url_override is provided, it is returned immediately without fetching."""
        override = "https://cdn.example.com/override_report.pdf"
        with patch("gam_registration.twdb_enrich.requests.get") as mock_get:
            result = twdb_enrich.discover_report_url(
                LANDING_PAGE_URL,
                html=None,
                report_url_override=override,
            )
        assert result == override
        mock_get.assert_not_called()

    def test_override_takes_precedence_over_html(self):
        """Override is returned even when html is also provided."""
        override = "https://cdn.example.com/override.pdf"
        result = twdb_enrich.discover_report_url(
            LANDING_PAGE_URL,
            html=LANDING_PAGE_HTML_WITH_REPORT,
            report_url_override=override,
        )
        assert result == override

    def test_empty_override_falls_through_to_discovery(self):
        """An empty or whitespace override does not prevent discovery from running."""
        result = twdb_enrich.discover_report_url(
            LANDING_PAGE_URL,
            html=LANDING_PAGE_HTML_WITH_REPORT,
            report_url_override="   ",  # whitespace — should be treated as absent
        )
        # Should discover the PDF from the HTML.
        assert result is not None
        assert result.endswith(".pdf") or "pdf" in result.lower()

    # --- Discovery found ---

    def test_finds_pdf_link_in_html(self):
        """discover_report_url returns a PDF URL when a report link exists in the HTML."""
        result = twdb_enrich.discover_report_url(
            LANDING_PAGE_URL,
            html=LANDING_PAGE_HTML_WITH_REPORT,
        )
        assert result is not None
        assert result.endswith("blsm_gam_report.pdf") or ".pdf" in result

    def test_resolves_relative_href_to_absolute_url(self):
        """Relative href is resolved against the landing page URL."""
        result = twdb_enrich.discover_report_url(
            LANDING_PAGE_URL,
            html=LANDING_PAGE_HTML_WITH_REPORT,
        )
        assert result is not None
        assert result.startswith("https://")

    def test_prefers_final_report_over_appendix(self):
        """When multiple PDFs exist, 'Final Report' text is preferred."""
        result = twdb_enrich.discover_report_url(
            LANDING_PAGE_URL,
            html=LANDING_PAGE_HTML_MULTIPLE_PDFS,
        )
        assert result is not None
        # The final_report_full.pdf link has higher score due to "Final Report" text.
        assert "final_report" in result or "report" in result.lower()

    # --- Discovery not found ---

    def test_returns_none_when_no_pdf_links(self):
        """Returns None when the HTML contains no PDF links."""
        result = twdb_enrich.discover_report_url(
            LANDING_PAGE_URL,
            html=LANDING_PAGE_HTML_NO_PDF,
        )
        assert result is None

    def test_returns_none_on_network_error(self):
        """Returns None (not an exception) when the landing page fetch fails."""
        with patch("gam_registration.twdb_enrich.requests.get", side_effect=ConnectionError("no network")):
            result = twdb_enrich.discover_report_url(LANDING_PAGE_URL)
        assert result is None

    # --- Fetch behavior when html not provided ---

    def test_fetches_landing_page_when_html_is_none(self):
        """When html=None, the landing page is fetched via requests.get."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = LANDING_PAGE_HTML_WITH_REPORT
        with patch("gam_registration.twdb_enrich.requests.get", return_value=mock_resp) as mock_get:
            result = twdb_enrich.discover_report_url(LANDING_PAGE_URL)
        mock_get.assert_called_once_with(LANDING_PAGE_URL, timeout=60)
        assert result is not None

    def test_does_not_fetch_when_html_provided(self):
        """When html is provided, requests.get is not called."""
        with patch("gam_registration.twdb_enrich.requests.get") as mock_get:
            twdb_enrich.discover_report_url(
                LANDING_PAGE_URL,
                html=LANDING_PAGE_HTML_WITH_REPORT,
            )
        mock_get.assert_not_called()


# ===========================================================================
# build_link_resources
# ===========================================================================

class TestBuildLinkResources:
    """Tests for build_link_resources."""

    _LANDING_URL = "https://www.twdb.texas.gov/gam/blsm.asp"
    _REPORT_URL = "https://www.twdb.texas.gov/gam/blsm_report.pdf"

    def test_returns_two_resources_when_both_urls_provided(self):
        """Returns a list with two resource dicts when both URLs are non-empty."""
        resources = twdb_enrich.build_link_resources(self._LANDING_URL, self._REPORT_URL)
        assert len(resources) == 2

    def test_landing_page_resource_shape(self):
        """Landing page resource has correct name, url, format, and url_type."""
        resources = twdb_enrich.build_link_resources(self._LANDING_URL, self._REPORT_URL)
        landing = next((r for r in resources if r["name"] == "twdb-landing-page"), None)
        assert landing is not None, "No 'twdb-landing-page' resource found"
        assert landing["url"] == self._LANDING_URL
        assert landing["format"].upper() == "HTML"
        assert landing["url_type"] == "url"
        assert "description" in landing

    def test_report_pdf_resource_shape(self):
        """Report PDF resource has correct name, url, format, and url_type."""
        resources = twdb_enrich.build_link_resources(self._LANDING_URL, self._REPORT_URL)
        report = next((r for r in resources if r["name"] == "gam-report-pdf"), None)
        assert report is not None, "No 'gam-report-pdf' resource found"
        assert report["url"] == self._REPORT_URL
        assert report["format"].upper() == "PDF"
        assert report["url_type"] == "url"
        assert "description" in report

    def test_omits_landing_page_resource_when_url_is_none(self):
        """If landing_page_url is None, no twdb-landing-page resource is returned."""
        resources = twdb_enrich.build_link_resources(None, self._REPORT_URL)
        names = {r["name"] for r in resources}
        assert "twdb-landing-page" not in names
        assert "gam-report-pdf" in names

    def test_omits_report_resource_when_url_is_none(self):
        """If report_url is None, no gam-report-pdf resource is returned."""
        resources = twdb_enrich.build_link_resources(self._LANDING_URL, None)
        names = {r["name"] for r in resources}
        assert "gam-report-pdf" not in names
        assert "twdb-landing-page" in names

    def test_returns_empty_list_when_both_urls_are_none(self):
        """Returns empty list when both URLs are None."""
        resources = twdb_enrich.build_link_resources(None, None)
        assert resources == []

    def test_returns_empty_list_when_both_urls_are_empty_strings(self):
        """Returns empty list when both URLs are empty strings."""
        resources = twdb_enrich.build_link_resources("", "")
        assert resources == []

    def test_no_byte_upload_fields(self):
        """Resource dicts must not contain a 'upload' or 'local_path' key."""
        resources = twdb_enrich.build_link_resources(self._LANDING_URL, self._REPORT_URL)
        for r in resources:
            assert "upload" not in r
            assert "local_path" not in r

    def test_url_values_are_stripped(self):
        """Whitespace is stripped from URL values."""
        resources = twdb_enrich.build_link_resources(
            "  " + self._LANDING_URL + "  ",
            "  " + self._REPORT_URL + "  ",
        )
        for r in resources:
            assert r["url"] == r["url"].strip()
