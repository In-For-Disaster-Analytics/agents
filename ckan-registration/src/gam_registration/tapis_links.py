"""Tapis v3 Files postit minting and CKAN url-type resource building.

Register-by-reference mode:
  Files stay on Corral/Tapis storage; a "postit" (redeemable HTTP URL) is
  minted for each file and registered as a CKAN url-type resource.
  This eliminates byte-uploads and avoids hammering CKAN with large file data.

Key functions
-------------
local_to_tapis_path:
    Convert an absolute local path to the Tapis file path relative to the
    storage-system root directory.

mint_postit_url:
    POST a postit for a single (system_id, tapis_path) pair and return the
    redeemable URL.

build_tapis_link_resources:
    Build the full list of CKAN url-type resource dicts for a resource_plan
    by minting a postit per file.

refresh_postit_urls:
    Re-mint postits for a list of (system_id, tapis_path) pairs.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retryable status codes (mirror ckan_action_post pattern).
# ---------------------------------------------------------------------------
_TAPIS_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def local_to_tapis_path(local_abs: "Path | str", system_root_dir: str) -> str:
    """Return the Tapis file path for *local_abs*, relative to *system_root_dir*.

    The result is POSIX-style with no leading '/'.

    Parameters
    ----------
    local_abs:
        Absolute local path to the file.
    system_root_dir:
        The Tapis storage system's ``rootDir`` (absolute local path on the
        storage host; the part that is NOT included in the Tapis path).

    Returns
    -------
    str
        Path relative to *system_root_dir*, e.g. ``"ygjk/Model_File/ygjk.nam"``.

    Raises
    ------
    ValueError
        If *local_abs* is not under *system_root_dir*.
    """
    local_path = Path(local_abs).resolve()
    root_path = Path(system_root_dir).resolve()

    try:
        rel = local_path.relative_to(root_path)
    except ValueError:
        raise ValueError(
            f"local_to_tapis_path: '{local_abs}' is not under system_root_dir '{system_root_dir}'."
        )

    # POSIX string, no leading slash.
    return rel.as_posix().lstrip("/")


# ---------------------------------------------------------------------------
# Postit minting
# ---------------------------------------------------------------------------

def mint_postit_url(
    system_id: str,
    tapis_path: str,
    *,
    base_url: str,
    jwt: str,
    allowed_uses: int = -1,
    valid_seconds: int = 3153600000,
    timeout: int = 60,
) -> str:
    """Mint a Tapis v3 postit for *system_id*/*tapis_path* and return its redeem URL.

    Calls:
        POST {base_url}/v3/files/postits/{system_id}/{tapis_path}
            ?allowedUses={allowed_uses}&validSeconds={valid_seconds}
        Authorization: Bearer {jwt}

    Response parsing:
        1. Prefer ``result.redeemUrl`` if present.
        2. Fall back to constructing ``{base_url}/v3/files/postits/redeem/{result.id}``.
        3. If neither is present, log the raw result and raise RuntimeError.

    Transient 429/5xx failures are retried with exponential back-off (mirrors
    the ``ckan_action_post`` retry pattern).

    Parameters
    ----------
    system_id:
        Tapis storage system ID (e.g. ``"corral-gam"``).
    tapis_path:
        File path on the system (e.g. ``"ygjk/Model_File/ygjk.nam"``).
    base_url:
        Tapis tenant base URL (e.g. ``"https://portals.tapis.io"``).
    jwt:
        Tapis bearer JWT.
    allowed_uses:
        Number of times the postit can be redeemed; -1 means unlimited.
    valid_seconds:
        Postit TTL in seconds (default ~100 years; tenant may cap this).
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    str
        The redeemable postit URL.
    """
    # Strip trailing slash from base and leading slash from path to avoid
    # double-slash in the constructed URL.
    base = base_url.rstrip("/")
    path_clean = tapis_path.lstrip("/")

    url = f"{base}/v3/files/postits/{system_id}/{path_clean}"
    params = {"allowedUses": allowed_uses, "validSeconds": valid_seconds}
    # Tapis canonical auth header is X-Tapis-Token; some gateways also accept
    # Authorization: Bearer. Send both so the pipeline matches whatever the
    # tenant requires (the unused one is ignored).
    headers = {
        "X-Tapis-Token": jwt,
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }

    max_retries = 5
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, params=params, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries:
                backoff = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 0.5)
                logger.warning(
                    "mint_postit_url: connection error (attempt %d/%d) for %s/%s: %s; "
                    "retrying in %.1fs",
                    attempt + 1, max_retries + 1, system_id, tapis_path, exc, backoff,
                )
                time.sleep(backoff)
                continue
            raise

        if resp.status_code in _TAPIS_RETRYABLE_STATUSES and attempt < max_retries:
            backoff = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 0.5)
            logger.warning(
                "mint_postit_url: HTTP %d (attempt %d/%d) for %s/%s; retrying in %.1fs",
                resp.status_code, attempt + 1, max_retries + 1, system_id, tapis_path, backoff,
            )
            time.sleep(backoff)
            continue

        # Non-retryable error.
        if resp.status_code >= 400:
            raise RuntimeError(
                f"mint_postit_url: Tapis postit request failed HTTP {resp.status_code} "
                f"for {system_id}/{tapis_path}: {resp.text[:500]}"
            )

        # Parse the successful response.
        try:
            body = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"mint_postit_url: invalid JSON response for {system_id}/{tapis_path}: {resp.text[:500]}"
            ) from exc

        result = body.get("result") or {}

        # Prefer the ready-made redeemUrl.
        redeem_url = result.get("redeemUrl") or result.get("redeem_url")
        if redeem_url:
            return str(redeem_url)

        # Fall back: construct from result.id.
        postit_id = result.get("id")
        if postit_id:
            return f"{base}/v3/files/postits/redeem/{postit_id}"

        # Neither present — log and fail.
        logger.error(
            "mint_postit_url: unexpected Tapis postit response for %s/%s — "
            "neither 'redeemUrl' nor 'id' found in result: %s",
            system_id, tapis_path, result,
        )
        raise RuntimeError(
            f"mint_postit_url: cannot determine redeem URL for {system_id}/{tapis_path}. "
            f"Tapis postit result did not contain 'redeemUrl' or 'id'. "
            f"Raw result: {result}"
        )

    # Should be unreachable.
    raise RuntimeError(
        f"mint_postit_url: retry loop exhausted for {system_id}/{tapis_path}"
    )


# ---------------------------------------------------------------------------
# Build CKAN url-type resource dicts from a resource_plan
# ---------------------------------------------------------------------------

def build_tapis_link_resources(
    resource_plan: list[dict[str, Any]],
    *,
    system_id: str,
    system_root_dir: str,
    base_url: str,
    jwt: str,
    allowed_uses: int = -1,
    valid_seconds: int = 3153600000,
) -> list[dict[str, Any]]:
    """Build CKAN url-type resource dicts by minting a Tapis postit per file.

    For each item in *resource_plan*, this function:
      1. Derives the Tapis file path using :func:`local_to_tapis_path`.
      2. Mints a postit URL via :func:`mint_postit_url`.
      3. Assembles a CKAN resource dict suitable for
         :func:`utils.create_link_resources`.

    On a per-file mint failure the file is skipped with a warning and the loop
    continues (collect-and-skip policy, not abort-on-first-error).

    Parameters
    ----------
    resource_plan:
        List of resource dicts as produced by ``utils.build_resource_plan``.
        Each item must have ``local_path`` (Path) and ``resource_name`` (str).
    system_id:
        Tapis storage system ID.
    system_root_dir:
        The system's ``rootDir`` on the local filesystem; used to compute the
        Tapis-relative path for each file.
    base_url:
        Tapis tenant base URL.
    jwt:
        Tapis bearer JWT.
    allowed_uses:
        ``allowedUses`` param for each postit (default -1 = unlimited).
    valid_seconds:
        ``validSeconds`` param for each postit.

    Returns
    -------
    list[dict]
        CKAN url-type resource dicts; items with mint failures are omitted.
        Each dict contains at minimum: ``resource_name``, ``name``, ``url``,
        ``description``, ``format``.  Any ``mint_*`` keys from the source item
        are passed through.
    """
    results: list[dict[str, Any]] = []
    throttle_delay = float(os.environ.get("CKAN_CALL_DELAY_SECONDS", "0.5"))
    failures: list[str] = []

    for i, item in enumerate(resource_plan):
        resource_name = str(item.get("resource_name") or "")
        local_path: Path = item["local_path"]

        # Derive tapis path.
        try:
            tapis_path = local_to_tapis_path(local_path, system_root_dir)
        except ValueError as exc:
            logger.warning(
                "build_tapis_link_resources: skipping '%s' — cannot compute Tapis path: %s",
                resource_name, exc,
            )
            failures.append(resource_name)
            continue

        # Throttle between mints to be gentle on Tapis.
        if i > 0 and throttle_delay > 0:
            time.sleep(throttle_delay)

        # Mint postit.
        try:
            postit_url = mint_postit_url(
                system_id,
                tapis_path,
                base_url=base_url,
                jwt=jwt,
                allowed_uses=allowed_uses,
                valid_seconds=valid_seconds,
            )
        except Exception as exc:
            logger.warning(
                "build_tapis_link_resources: skipping '%s' — mint failed: %s",
                resource_name, exc,
            )
            failures.append(resource_name)
            continue

        # Determine format from suffix if not already set.
        fmt = item.get("format")
        if not fmt:
            suffix = local_path.suffix.lower().lstrip(".")
            fmt = suffix.upper() if suffix else "BIN"

        resource_dict: dict[str, Any] = {
            "resource_name": resource_name,
            "name": resource_name,
            "url": postit_url,
            "description": str(item.get("resource_description") or ""),
            "format": fmt,
        }

        # Pass through any mint_ prefixed keys from the source item.
        for key, value in item.items():
            if key.startswith("mint_"):
                resource_dict[key] = value

        results.append(resource_dict)

    if failures:
        logger.warning(
            "build_tapis_link_resources: %d file(s) skipped due to mint failures: %s",
            len(failures), failures,
        )

    return results


# ---------------------------------------------------------------------------
# Refresh expired postits
# ---------------------------------------------------------------------------

def refresh_postit_urls(
    file_pairs: list[tuple[str, str]],
    *,
    base_url: str,
    jwt: str,
    allowed_uses: int = -1,
    valid_seconds: int = 3153600000,
) -> list[str]:
    """Re-mint postits for a list of (system_id, tapis_path) pairs.

    Use this when postits have expired because the Tapis tenant has capped
    ``validSeconds`` below the value used when the resources were first
    registered.  Returns fresh redeem URLs in the same order as *file_pairs*.
    On per-file failure, ``None`` is returned in that position and a warning
    is logged.

    Parameters
    ----------
    file_pairs:
        List of ``(system_id, tapis_path)`` tuples.
    base_url:
        Tapis tenant base URL.
    jwt:
        Fresh Tapis JWT.
    allowed_uses:
        ``allowedUses`` for each new postit.
    valid_seconds:
        ``validSeconds`` for each new postit.

    Returns
    -------
    list[str | None]
        Fresh redeem URLs, or ``None`` where minting failed.
    """
    fresh_urls: list[str | None] = []
    throttle_delay = float(os.environ.get("CKAN_CALL_DELAY_SECONDS", "0.5"))

    for i, (system_id, tapis_path) in enumerate(file_pairs):
        if i > 0 and throttle_delay > 0:
            time.sleep(throttle_delay)
        try:
            url = mint_postit_url(
                system_id,
                tapis_path,
                base_url=base_url,
                jwt=jwt,
                allowed_uses=allowed_uses,
                valid_seconds=valid_seconds,
            )
            fresh_urls.append(url)
        except Exception as exc:
            logger.warning(
                "refresh_postit_urls: failed to re-mint postit for %s/%s: %s",
                system_id, tapis_path, exc,
            )
            fresh_urls.append(None)

    return fresh_urls
