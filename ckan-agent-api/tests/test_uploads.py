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


def test_serve_upload_file(tmp_path: Path):
    client = _client(tmp_path)
    resp = client.post("/v1/uploads", files=[("files", ("data.txt", b"hello world", "text/plain"))])
    body = resp.json()
    upload_id = body["upload_id"]
    serve = client.get(f"/v1/uploads/{upload_id}/data.txt")
    assert serve.status_code == 200
    assert serve.content == b"hello world"


def test_serve_upload_invalid_id(tmp_path: Path):
    client = _client(tmp_path)
    resp = client.get("/v1/uploads/not-a-uuid/file.txt")
    assert resp.status_code == 404


def test_serve_upload_cross_id_isolation(tmp_path: Path):
    client = _client(tmp_path)
    r1 = client.post("/v1/uploads", files=[("files", ("a.txt", b"data-A", "text/plain"))])
    r2 = client.post("/v1/uploads", files=[("files", ("b.txt", b"data-B", "text/plain"))])
    uid1 = r1.json()["upload_id"]
    uid2 = r2.json()["upload_id"]
    # Trying to reach uid2's file under uid1's namespace must 404
    assert client.get(f"/v1/uploads/{uid1}/{uid2}/b.txt").status_code == 404
    # But each upload can serve its own file
    assert client.get(f"/v1/uploads/{uid1}/a.txt").status_code == 200
    assert client.get(f"/v1/uploads/{uid2}/b.txt").status_code == 200


def test_serve_upload_missing_file(tmp_path: Path):
    client = _client(tmp_path)
    resp = client.post("/v1/uploads", files=[("files", ("a.txt", b"x", "text/plain"))])
    upload_id = resp.json()["upload_id"]
    resp2 = client.get(f"/v1/uploads/{upload_id}/nonexistent.bin")
    assert resp2.status_code == 404
