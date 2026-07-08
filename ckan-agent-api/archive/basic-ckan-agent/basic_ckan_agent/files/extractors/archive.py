from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def inspect_zip(path: Path, *, max_members: int) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        return {"is_zipfile": False, "message": "File is not a readable ZIP archive."}

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        members = [
            {
                "name": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "is_dir": info.is_dir(),
            }
            for info in infos[:max_members]
        ]

    return {
        "is_zipfile": True,
        "member_count": len(infos),
        "members_truncated": len(infos) > max_members,
        "total_uncompressed_size": sum(info.file_size for info in infos),
        "members": members,
        "shapefile_candidates": _shapefile_candidates(infos),
    }


def _shapefile_candidates(infos: list[zipfile.ZipInfo]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for info in infos:
        name = Path(info.filename)
        suffix = name.suffix.lower()
        if suffix in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
            grouped[name.with_suffix("").as_posix()].add(suffix)

    candidates: list[dict[str, Any]] = []
    for stem, suffixes in sorted(grouped.items()):
        candidates.append(
            {
                "stem": stem,
                "components": sorted(suffixes),
                "has_required_components": {".shp", ".shx", ".dbf"}.issubset(suffixes),
                "has_projection": ".prj" in suffixes,
            }
        )
    return candidates
