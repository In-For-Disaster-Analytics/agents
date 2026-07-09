"""File/zip upload endpoint (spec Increment 8a).

Accepts multipart uploads, stores them under a per-upload directory in ``upload_root``, safely
extracts any ``.zip`` (zip-bomb / path-traversal / symlink guards), and returns a head inventory
the chat/runs call can reference via ``upload_dir``. No CKAN writes happen here.
"""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.files import ArchiveError, build_head_inventory, safe_extract_zip
from app.settings import Settings, get_settings

router = APIRouter(tags=["uploads"])

_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _safe_name(name: str | None) -> str:
    base = Path(str(name or "upload")).name
    cleaned = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", base).strip() or "upload"
    return cleaned[:200]


def _save_within_limit(upload: UploadFile, target: Path, max_bytes: int) -> int:
    written = 0
    with target.open("wb") as out:
        while True:
            chunk = upload.file.read(1 << 16)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out.close()
                target.unlink(missing_ok=True)
                raise ValueError(f"Upload {target.name!r} exceeds the {max_bytes}-byte limit.")
            out.write(chunk)
    return written


@router.post("/v1/uploads", operation_id="createUpload")
def create_upload(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    upload_id = uuid.uuid4().hex
    dest = (settings.upload_root / upload_id).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    stored: list[Path] = []

    for upload in files:
        target = dest / _safe_name(upload.filename)
        try:
            _save_within_limit(upload, target, settings.max_upload_bytes)
        except ValueError as exc:
            warnings.append(str(exc))
            continue

        if target.suffix.lower() == ".zip":
            try:
                extracted = safe_extract_zip(
                    target,
                    dest,
                    max_uncompressed=settings.max_zip_uncompressed_bytes,
                    max_members=settings.max_zip_members,
                    max_file_bytes=settings.max_file_bytes,
                )
                stored.extend(extracted)
            except ArchiveError as exc:
                warnings.append(f"{target.name}: {exc}")
            finally:
                target.unlink(missing_ok=True)  # don't keep/serve the raw zip
        else:
            stored.append(target)

    inventory = build_head_inventory(stored)
    return {
        "ok": True,
        "upload_id": upload_id,
        "dir": str(dest),
        "file_count": len(inventory),
        "files": inventory,
        "warnings": warnings,
    }


@router.get("/v1/uploads/{upload_id}/{path:path}", operation_id="getUploadFile")
def get_upload_file(
    upload_id: str,
    path: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve a previously uploaded file by upload_id and relative path.

    Auth: the upload_id (UUID hex) is unguessable and acts as a bearer token.
    Abaco /vsicurl/ reads use this to fetch spatial files without a CKAN round-trip.
    """
    if not _UPLOAD_ID_RE.match(upload_id):
        raise HTTPException(status_code=404, detail="Not found")
    upload_dir = (settings.upload_root / upload_id).resolve()
    target = (upload_dir / path).resolve()
    try:
        target.relative_to(upload_dir)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(target))
