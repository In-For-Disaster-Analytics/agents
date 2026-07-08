from __future__ import annotations

from pydantic import BaseModel, Field


class FilePathInput(BaseModel):
    path: str = Field(description="Local file path exactly as supplied by the user.")


class TextFileInput(FilePathInput):
    max_chars: int = Field(
        default=8000,
        ge=1,
        le=50000,
        description="Maximum number of text characters to return.",
    )


class PdfTextInput(FilePathInput):
    page_start: int = Field(default=0, ge=0, description="Zero-based page index to start reading.")
    max_pages: int = Field(default=5, ge=1, le=20, description="Maximum number of pages to inspect.")
    max_chars: int = Field(default=12000, ge=1, le=50000, description="Maximum extracted characters to return.")


class CsvProfileInput(FilePathInput):
    max_rows: int = Field(default=25, ge=1, le=200, description="Maximum data rows to return as examples.")


class JsonProfileInput(FilePathInput):
    max_sample_chars: int = Field(
        default=12000,
        ge=1000,
        le=50000,
        description="Maximum characters to retain in compact JSON samples.",
    )


class ZipInspectInput(FilePathInput):
    max_members: int = Field(default=100, ge=1, le=500, description="Maximum archive members to list.")
