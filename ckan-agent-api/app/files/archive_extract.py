"""Safe ZIP extraction (spec Increment 8a).

Guards against the classic archive attacks before/while extracting:
- zip-bomb: reject if the declared OR actual cumulative uncompressed size exceeds a cap;
- too-many-members: reject above a member-count cap;
- per-file size: reject members larger than a cap (declared and while streaming);
- path-traversal: reject members with absolute paths or ``..`` components, and verify every
  resolved target stays inside the destination directory;
- symlinks: reject symlink members.

Never trusts the declared sizes alone — extraction streams with a running cap so a lying header
cannot write past the limit.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

_CHUNK = 1 << 16
_SYMLINK_MODE = 0o120000


class ArchiveError(RuntimeError):
    """Raised when an archive is unsafe or cannot be extracted within limits."""


def _is_unsafe_member_name(name: str) -> bool:
    if not name or name.startswith("/") or name.startswith("\\"):
        return True
    parts = Path(name).parts
    return ".." in parts or any(p.startswith("/") for p in parts)


def safe_extract_zip(
    zip_path: Path,
    dest_dir: Path,
    *,
    max_uncompressed: int,
    max_members: int,
    max_file_bytes: int,
) -> list[Path]:
    """Extract *zip_path* into *dest_dir* safely. Returns the list of extracted file paths."""
    if not zipfile.is_zipfile(zip_path):
        raise ArchiveError(f"Not a valid ZIP archive: {zip_path.name}")

    dest = dest_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total_written = 0

    with zipfile.ZipFile(zip_path) as archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        if len(infos) > max_members:
            raise ArchiveError(f"ZIP has {len(infos)} members, exceeding the {max_members} limit.")
        declared_total = sum(i.file_size for i in infos)
        if declared_total > max_uncompressed:
            raise ArchiveError(
                f"ZIP declares {declared_total} uncompressed bytes, exceeding the {max_uncompressed} limit (possible zip bomb)."
            )

        for info in infos:
            if _is_unsafe_member_name(info.filename):
                raise ArchiveError(f"Refusing unsafe archive member path: {info.filename!r}")
            if ((info.external_attr >> 16) & 0o170000) == _SYMLINK_MODE:
                raise ArchiveError(f"Refusing symlink archive member: {info.filename!r}")
            if info.file_size > max_file_bytes:
                raise ArchiveError(f"Archive member {info.filename!r} exceeds the {max_file_bytes}-byte per-file limit.")

            target = (dest / info.filename).resolve()
            if target != dest and dest not in target.parents:
                raise ArchiveError(f"Archive member escapes the extraction directory: {info.filename!r}")

            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(info) as src, target.open("wb") as out:
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    total_written += len(chunk)
                    if written > max_file_bytes:
                        raise ArchiveError(f"Archive member {info.filename!r} exceeded the per-file limit while extracting.")
                    if total_written > max_uncompressed:
                        raise ArchiveError("Archive exceeded the total uncompressed limit while extracting (possible zip bomb).")
                    out.write(chunk)
            extracted.append(target)

    return extracted
