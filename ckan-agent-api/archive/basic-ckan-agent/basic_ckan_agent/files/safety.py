from __future__ import annotations

import mimetypes
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from basic_ckan_agent.settings import env

DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
SUPPORTED_PATH_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".xml",
    ".html",
    ".htm",
    ".pdf",
    ".csv",
    ".tsv",
    ".json",
    ".geojson",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".tif",
    ".tiff",
    ".zip",
    ".shp",
}
SENSITIVE_FILE_NAMES = {
    ".env",
    ".env.local",
    ".envrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt", ".cer"}


class FileSafetyError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SafeFile:
    path: Path
    requested_path: str
    size_bytes: int
    mime_type: str
    extension: str


def extract_file_paths(message: str) -> list[str]:
    candidates: list[str] = []
    for quoted in re.findall(r"""["'`]([^"'`]+)["'`]""", message):
        if _looks_like_path(quoted):
            candidates.append(quoted)

    for token in re.split(r"\s+", message):
        cleaned = token.strip().strip(",;:()[]{}<>")
        if not cleaned or cleaned in candidates:
            continue
        if _looks_like_path(cleaned):
            candidates.append(cleaned)

    seen: set[str] = set()
    paths: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        paths.append(candidate)
    return paths


def normalize_allowed_paths(paths: Iterable[str] | None, *, base_dir: Path | None = None) -> set[str] | None:
    if paths is None:
        return None
    return {_candidate_path(path, base_dir=base_dir).resolve(strict=False).as_posix() for path in paths}


def resolve_user_path(
    path: str,
    *,
    allowed_paths: set[str] | None = None,
    base_dir: Path | None = None,
) -> Path:
    candidate = _candidate_path(path, base_dir=base_dir)
    resolved = candidate.resolve(strict=False)
    if allowed_paths is not None and resolved.as_posix() not in allowed_paths:
        raise FileSafetyError(
            "path_not_allowed",
            "This file tool can only inspect paths explicitly mentioned in the current user request.",
        )
    return resolved


def validate_readable_file(
    path: str,
    *,
    allowed_paths: set[str] | None = None,
    max_bytes: int | None = None,
    base_dir: Path | None = None,
) -> SafeFile:
    resolved = resolve_user_path(path, allowed_paths=allowed_paths, base_dir=base_dir)
    if not resolved.exists():
        raise FileSafetyError("not_found", f"File does not exist: {path}")
    if not resolved.is_file():
        raise FileSafetyError("not_a_file", f"Path is not a regular file: {path}")
    _reject_sensitive_path(resolved)

    size_bytes = resolved.stat().st_size
    byte_limit = max_bytes if max_bytes is not None else max_file_bytes()
    if size_bytes > byte_limit:
        raise FileSafetyError(
            "file_too_large",
            f"File is {size_bytes} bytes, which exceeds the {byte_limit} byte read limit.",
        )
    if not os.access(resolved, os.R_OK):
        raise FileSafetyError("not_readable", f"File is not readable: {path}")

    mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return SafeFile(
        path=resolved,
        requested_path=path,
        size_bytes=size_bytes,
        mime_type=mime_type,
        extension=resolved.suffix.lower(),
    )


def max_file_bytes() -> int:
    raw = env("CKAN_AGENT_MAX_FILE_BYTES")
    if not raw:
        return DEFAULT_MAX_FILE_BYTES
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_FILE_BYTES


def _candidate_path(path: str, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir or Path.cwd()) / candidate
    return candidate


def _looks_like_path(value: str) -> bool:
    if "://" in value:
        return False
    candidate = value.strip()
    if not candidate:
        return False
    if candidate.startswith(("/", "~/", "./", "../")):
        return True
    suffix = Path(candidate).suffix.lower()
    return suffix in SUPPORTED_PATH_SUFFIXES


def _reject_sensitive_path(path: Path) -> None:
    names = {part for part in path.parts}
    if names & SENSITIVE_FILE_NAMES or ".git" in names or ".ssh" in names:
        raise FileSafetyError("sensitive_path", "Refusing to inspect a sensitive local path.")
    if path.suffix.lower() in SENSITIVE_SUFFIXES:
        raise FileSafetyError("sensitive_path", "Refusing to inspect a file with a sensitive credential suffix.")
