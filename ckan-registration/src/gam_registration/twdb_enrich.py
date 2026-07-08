"""TWDB landing-page enrichment: report PDF discovery and CKAN link resources.

This module handles:
  - B1: discover_report_url — find the GAM report PDF link from a TWDB landing page.
  - B4: build_link_resources — produce url-type CKAN resource dicts for the landing
    page and report PDF (no byte copy).

Manifest `report_url` override takes precedence over auto-discovery in all cases.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

# Keywords (case-insensitive) that suggest a link is the primary report PDF.
# Ordered by specificity — earlier entries score higher.
_REPORT_KEYWORDS: list[str] = [
    "final report",
    "model report",
    "technical report",
    "gam report",
    "groundwater availability model",
    "report",
    "executive summary",
    "model documentation",
]

_PDF_EXTENSION_RE = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)


def _score_link(href: str, link_text: str, base_url: str) -> int:
    """Score a single anchor href + text for report-PDF likelihood.

    Returns a non-negative integer; higher is better.
    Returns -1 to hard-exclude the link.
    """
    href_lower = href.lower()
    text_lower = link_text.lower().strip()

    # Must end in .pdf (with optional query string) or contain "pdf" in the path.
    is_pdf_href = bool(_PDF_EXTENSION_RE.search(href_lower))
    has_pdf_token = "pdf" in href_lower

    if not (is_pdf_href or has_pdf_token):
        return -1  # Exclude: not a PDF link.

    score = 0
    if is_pdf_href:
        score += 5  # Explicit .pdf extension is a strong signal.

    for i, keyword in enumerate(_REPORT_KEYWORDS):
        if keyword in text_lower:
            score += len(_REPORT_KEYWORDS) - i + 3  # Earlier keywords score more.
            break  # Only the best keyword match counts.

    for i, keyword in enumerate(_REPORT_KEYWORDS):
        if keyword in href_lower:
            score += len(_REPORT_KEYWORDS) - i + 1
            break

    return score


def _resolve_url(href: str, base_url: str) -> str:
    """Resolve a potentially relative href against base_url."""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return urljoin(base_url, href)


def discover_report_url(
    landing_page_url: str,
    html: str | None = None,
    *,
    report_url_override: str | None = None,
    timeout: int = 60,
) -> str | None:
    """Discover the GAM report PDF URL from a TWDB landing page.

    Two-layer precedence:
    1. If *report_url_override* is provided and non-empty, return it immediately
       without fetching the landing page.
    2. Otherwise, fetch the landing page (or use the provided *html*), parse
       ``<a>`` tags, score candidates, and return the best match.

    Parameters
    ----------
    landing_page_url:
        URL of the TWDB landing page (e.g. ``https://www.twdb.texas.gov/...``).
    html:
        Pre-fetched HTML string.  If provided, the landing page is not fetched
        again.  Pass ``None`` to fetch automatically.
    report_url_override:
        If non-empty, this value is returned immediately and takes precedence
        over any auto-discovery result.  Corresponds to the manifest ``report_url``
        field.
    timeout:
        HTTP request timeout in seconds (used only when *html* is None).

    Returns
    -------
    str or None
        Absolute URL of the report PDF, or ``None`` if none was found.
    """
    # Layer 1: manifest override takes precedence.
    if report_url_override and report_url_override.strip():
        logger.debug("twdb_enrich: using report_url override: %s", report_url_override)
        return report_url_override.strip()

    # Layer 2: auto-discover from landing page HTML.
    if html is None:
        try:
            response = requests.get(landing_page_url, timeout=timeout)
            response.raise_for_status()
            html = response.text
        except Exception as exc:
            logger.warning(
                "twdb_enrich: failed to fetch landing page %s: %s",
                landing_page_url,
                exc,
            )
            return None

    # Parse all <a href="..."> tags.
    anchor_re = re.compile(
        r'<a\b[^>]*\bhref=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    # Also match href before other attributes.
    anchor_re2 = re.compile(
        r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>',
        re.IGNORECASE,
    )

    candidates: list[tuple[int, str]] = []  # (score, absolute_url)

    for match in anchor_re.finditer(html):
        href = match.group(1).strip()
        link_text = re.sub(r"<[^>]+>", " ", match.group(2)).strip()
        if not href:
            continue
        abs_url = _resolve_url(href, landing_page_url)
        score = _score_link(href, link_text, landing_page_url)
        if score >= 0:
            candidates.append((score, abs_url))
            logger.debug(
                "twdb_enrich: candidate PDF link score=%d url=%s text=%r",
                score,
                abs_url,
                link_text[:80],
            )

    if not candidates:
        logger.info(
            "twdb_enrich: no PDF report link found on %s", landing_page_url
        )
        return None

    # Return the highest-scoring candidate.
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_url = candidates[0]
    logger.info(
        "twdb_enrich: selected report PDF (score=%d): %s", best_score, best_url
    )
    return best_url


def build_link_resources(
    landing_page_url: str | None,
    report_url: str | None,
) -> list[dict[str, Any]]:
    """Build url-type CKAN resource dicts for the landing page and report PDF.

    These are registered as ``url_type="url"`` resources — no byte copy is made.
    The caller passes these dicts to ``create_link_resources`` (defined in utils.py)
    or directly to ``ckan_action_post("resource_create", ...)``.

    Parameters
    ----------
    landing_page_url:
        TWDB landing page URL (from the manifest ``twdb_page_url`` field).
    report_url:
        GAM report PDF URL (from manifest override or auto-discovery).

    Returns
    -------
    list[dict]
        List of CKAN resource-dict skeletons, each with keys:
        ``name``, ``url``, ``format``, ``url_type``, ``description``.
        Only non-None, non-empty URLs are included.
    """
    resources: list[dict[str, Any]] = []

    if landing_page_url and landing_page_url.strip():
        resources.append(
            {
                "name": "twdb-landing-page",
                "url": landing_page_url.strip(),
                "format": "HTML",
                "url_type": "url",
                "description": (
                    "Texas Water Development Board (TWDB) official landing page "
                    "for this Groundwater Availability Model (GAM)."
                ),
            }
        )

    if report_url and report_url.strip():
        resources.append(
            {
                "name": "gam-report-pdf",
                "url": report_url.strip(),
                "format": "PDF",
                "url_type": "url",
                "description": (
                    "GAM technical report PDF published by the Texas Water "
                    "Development Board (TWDB)."
                ),
            }
        )

    return resources
