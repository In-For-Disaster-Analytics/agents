from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def inspect_image(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(64)

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", header[16:24])
        return {"image_format": "PNG", "width": width, "height": height}

    if header[:6] in {b"GIF87a", b"GIF89a"}:
        width, height = struct.unpack("<HH", header[6:10])
        return {"image_format": "GIF", "width": width, "height": height}

    if header.startswith(b"\xff\xd8"):
        dimensions = _jpeg_dimensions(path)
        return {"image_format": "JPEG", **dimensions}

    return {
        "image_format": None,
        "message": "Image dimensions could not be read from PNG, GIF, or JPEG headers.",
    }


def _jpeg_dimensions(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.read(2)
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if marker in {
                b"\xc0",
                b"\xc1",
                b"\xc2",
                b"\xc3",
                b"\xc5",
                b"\xc6",
                b"\xc7",
                b"\xc9",
                b"\xca",
                b"\xcb",
                b"\xcd",
                b"\xce",
                b"\xcf",
            }:
                segment = handle.read(segment_length - 2)
                if len(segment) >= 5:
                    precision = segment[0]
                    height, width = struct.unpack(">HH", segment[1:5])
                    return {"width": width, "height": height, "precision": precision}
                break
            handle.seek(segment_length - 2, 1)
    return {"width": None, "height": None}
