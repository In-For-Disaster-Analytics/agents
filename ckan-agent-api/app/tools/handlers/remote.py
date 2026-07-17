"""Remote resource tool handlers — authenticated URL fetches."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import requests

from app.auth_context import get_request_ckan_auth
from app.files.extractors.pdf import extract_pdf_text

_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def fetch_remote_pdf(args: dict[str, Any]) -> dict[str, Any]:
    """Download an authenticated remote PDF by URL and extract its text."""
    url = str(args["url"]).strip()
    if not url.startswith("https://"):
        return {"error": "invalid_url", "message": "Only HTTPS URLs are supported."}

    page_start = int(args.get("page_start", 0))
    max_pages = int(args.get("max_pages", 20))
    max_chars = int(args.get("max_chars", 8000))

    auth = get_request_ckan_auth()
    headers = {"Authorization": auth} if auth else {}

    try:
        resp = requests.get(url, headers=headers, timeout=30, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"error": "download_failed", "message": str(exc)}

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65_536):
            downloaded += len(chunk)
            if downloaded > _MAX_DOWNLOAD_BYTES:
                tmp_path.unlink(missing_ok=True)
                return {
                    "error": "file_too_large",
                    "message": f"PDF exceeds {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB limit.",
                }
            tmp.write(chunk)

    try:
        result = extract_pdf_text(tmp_path, page_start=page_start, max_pages=max_pages, max_chars=max_chars)
        result["source_url"] = url
        return result
    finally:
        tmp_path.unlink(missing_ok=True)
