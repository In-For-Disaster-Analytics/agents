from __future__ import annotations

import dataclasses
import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import get_settings


def _zip_bytes(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()


def _client(tmp_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: dataclasses.replace(
        get_settings(), upload_root=tmp_path / "uploads"
    )
    return TestClient(app)


def test_upload_zip_extracts_and_lists(tmp_path: Path):
    client = _client(tmp_path)
    zbytes = _zip_bytes({"a.txt": "hello", "data/b.csv": "x,y\n1,2\n"})
    resp = client.post("/v1/uploads", files=[("files", ("bundle.zip", zbytes, "application/zip"))])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {f["name"] for f in body["files"]}
    assert {"a.txt", "b.csv"} <= names
    assert "bundle.zip" not in names  # raw zip not kept/listed
    assert body["dir"].endswith(body["upload_id"])


def test_upload_plain_file(tmp_path: Path):
    client = _client(tmp_path)
    resp = client.post("/v1/uploads", files=[("files", ("notes.txt", b"some text", "text/plain"))])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["file_count"] == 1
    assert body["files"][0]["name"] == "notes.txt"
