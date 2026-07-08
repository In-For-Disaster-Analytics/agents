from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.files import build_head_inventory
from app.files.archive_extract import ArchiveError, safe_extract_zip

CAPS = {"max_uncompressed": 10_000_000, "max_members": 100, "max_file_bytes": 5_000_000}


def _zip(tmp_path: Path, entries: list[tuple], name: str = "a.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for entry in entries:
            if isinstance(entry, zipfile.ZipInfo):
                z.writestr(entry, "x")
            else:
                z.writestr(entry[0], entry[1])
    return path


def test_extracts_benign_zip(tmp_path: Path):
    zpath = _zip(tmp_path, [("a.txt", "hello"), ("sub/b.csv", "x,y\n1,2\n")])
    dest = tmp_path / "out"
    files = safe_extract_zip(zpath, dest, **CAPS)
    rels = sorted(str(p.relative_to(dest.resolve())) for p in files)
    assert rels == ["a.txt", "sub/b.csv"]
    assert all(dest.resolve() in p.parents for p in files)


def test_rejects_path_traversal(tmp_path: Path):
    zpath = _zip(tmp_path, [("../evil.txt", "x")])
    with pytest.raises(ArchiveError, match="unsafe archive member"):
        safe_extract_zip(zpath, tmp_path / "out", **CAPS)


def test_rejects_too_many_members(tmp_path: Path):
    zpath = _zip(tmp_path, [("a.txt", "1"), ("b.txt", "2")])
    with pytest.raises(ArchiveError, match="exceeding the 1 limit"):
        safe_extract_zip(zpath, tmp_path / "out", max_uncompressed=10_000, max_members=1, max_file_bytes=10_000)


def test_rejects_zip_bomb_declared(tmp_path: Path):
    zpath = _zip(tmp_path, [("big.txt", "abcdefghij")])
    with pytest.raises(ArchiveError, match="zip bomb"):
        safe_extract_zip(zpath, tmp_path / "out", max_uncompressed=5, max_members=100, max_file_bytes=10_000)


def test_rejects_symlink_member(tmp_path: Path):
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16  # symlink mode bits
    zpath = _zip(tmp_path, [info])
    with pytest.raises(ArchiveError, match="symlink"):
        safe_extract_zip(zpath, tmp_path / "out", **CAPS)


def test_build_head_inventory(tmp_path: Path):
    (tmp_path / "a.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02\x03")
    inv = build_head_inventory([tmp_path / "a.csv", tmp_path / "b.bin"])
    by_name = {e["name"]: e for e in inv}
    assert by_name["a.csv"]["kind"] == "tabular"
    assert "col1,col2" in by_name["a.csv"]["head"]
    assert by_name["b.bin"]["kind"] == "binary"
    assert "head" not in by_name["b.bin"]  # binary -> no text head


def _zip_bytes(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()
