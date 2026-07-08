from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def profile_csv(path: Path, *, max_rows: int) -> dict[str, Any]:
    sample_text = path.read_text(encoding="utf-8", errors="replace")[:8192]
    dialect = _sniff_dialect(sample_text)
    has_header = _sniff_has_header(sample_text)

    rows: list[list[str]] = []
    scanned_rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, dialect)
        for row in reader:
            scanned_rows += 1
            if len(rows) < max_rows + 1:
                rows.append(row)
            if scanned_rows >= 10000:
                break

    if not rows:
        return {
            "delimiter": dialect.delimiter,
            "has_header": has_header,
            "headers": [],
            "sample_rows": [],
            "row_count_sampled": 0,
            "column_count": 0,
        }

    headers = rows[0] if has_header else [f"column_{idx + 1}" for idx in range(max(len(row) for row in rows))]
    data_rows = rows[1:] if has_header else rows
    sample_rows = [_row_to_dict(headers, row) for row in data_rows[:max_rows]]

    return {
        "delimiter": dialect.delimiter,
        "quotechar": dialect.quotechar,
        "has_header": has_header,
        "headers": headers,
        "column_count": len(headers),
        "sample_rows": sample_rows,
        "row_count_sampled": max(0, scanned_rows - (1 if has_header else 0)),
        "row_count_truncated": scanned_rows >= 10000,
        "columns": _column_summaries(headers, data_rows),
    }


def _sniff_dialect(sample_text: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample_text)
    except csv.Error:
        return csv.get_dialect("excel")


def _sniff_has_header(sample_text: str) -> bool:
    try:
        return bool(csv.Sniffer().has_header(sample_text))
    except csv.Error:
        return True


def _row_to_dict(headers: list[str], row: list[str]) -> dict[str, str]:
    return {header: row[idx] if idx < len(row) else "" for idx, header in enumerate(headers)}


def _column_summaries(headers: list[str], data_rows: list[list[str]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for idx, header in enumerate(headers[:100]):
        values = [row[idx] for row in data_rows if idx < len(row) and row[idx] != ""]
        summaries.append(
            {
                "name": header,
                "non_empty_sample_count": len(values),
                "sample_values": _unique_prefix(values, limit=5),
            }
        )
    return summaries


def _unique_prefix(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result
