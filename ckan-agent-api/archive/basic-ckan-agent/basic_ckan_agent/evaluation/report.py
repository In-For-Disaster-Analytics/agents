"""Render eval results to a self-contained local HTML report.

No external assets, no network, no account: inline CSS only, opens in any
browser. ``render_html`` takes the list of experiment report dicts produced by
``run_experiments.run_experiment`` and returns a complete HTML document.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_STATUS_COLORS = {"pass": "#1a7f37", "review": "#9a6700", "fail": "#cf222e"}
_STATUS_BG = {"pass": "#dafbe1", "review": "#fff8c5", "fail": "#ffebe9"}


def render_html(reports: list[dict]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [_HEAD, f"<h1>CKAN Agent Evaluation Report</h1>", f"<p class='muted'>Generated {generated}</p>"]
    parts.append(_render_matrix(reports))
    for report in reports:
        parts.append(_render_experiment(report))
    parts.append("</body></html>")
    return "\n".join(parts)


def write_html_report(reports: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(reports), encoding="utf-8")
    return path


def _render_matrix(reports: list[dict]) -> str:
    rows = []
    for report in reports:
        md = report.get("metadata", {})
        s = report.get("summary", {})
        total = sum(s.get(k, 0) for k in ("pass", "review", "fail")) or 1
        pass_pct = round(100 * s.get("pass", 0) / total)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(md.get('prompt', '')))}</td>"
            f"<td>{html.escape(str(md.get('model', '')))}</td>"
            f"<td class='num'>{s.get('pass', 0)}</td>"
            f"<td class='num'>{s.get('review', 0)}</td>"
            f"<td class='num'>{s.get('fail', 0)}</td>"
            f"<td class='num'><b>{pass_pct}%</b></td>"
            f"<td class='muted'>{html.escape(str(md.get('graph_version', '')))}</td>"
            "</tr>"
        )
    return (
        "<h2>Matrix summary</h2>"
        "<table><thead><tr>"
        "<th>Prompt</th><th>Model</th><th>Pass</th><th>Review</th><th>Fail</th><th>Pass %</th><th>Commit</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_experiment(report: dict) -> str:
    md = report.get("metadata", {})
    s = report.get("summary", {})
    title = f"{md.get('prompt', '')} &middot; {md.get('model', '')}"
    header = (
        f"<h2>{html.escape(md.get('prompt', ''))} "
        f"<span class='muted'>/ {html.escape(md.get('model', ''))}</span></h2>"
        f"<p class='muted'>pass {s.get('pass', 0)} &middot; review {s.get('review', 0)} &middot; "
        f"fail {s.get('fail', 0)} &middot; {html.escape(str(md.get('run_date', '')))} &middot; "
        f"commit {html.escape(str(md.get('graph_version', '')))}</p>"
    )
    rows = [_render_example_row(r) for r in report.get("results", [])]
    table = (
        "<table><thead><tr>"
        "<th>Example</th><th>Type</th><th>Status</th><th>Title</th><th>Desc</th>"
        "<th>Faithful</th><th>Tools</th><th>Safe</th><th></th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return header + table


def _render_example_row(r: dict) -> str:
    fb = r.get("feedback", {})
    status = r.get("status", "")
    badge = (
        f"<span class='badge' style='color:{_STATUS_COLORS.get(status, '#444')};"
        f"background:{_STATUS_BG.get(status, '#eee')}'>{html.escape(status.upper())}</span>"
    )
    detail_id = html.escape(str(r.get("example_id", "")))
    main = (
        "<tr>"
        f"<td><code>{detail_id}</code></td>"
        f"<td class='muted'>{html.escape(str(r.get('task_type', '')))}</td>"
        f"<td>{badge}</td>"
        f"<td class='num'>{_fmt(fb.get('title_score'))}</td>"
        f"<td class='num'>{_fmt(fb.get('description_score'))}</td>"
        f"<td>{_yn(fb.get('faithfulness_pass'))}</td>"
        f"<td>{_yn(fb.get('correct_tool_called'))}</td>"
        f"<td>{_yn(fb.get('no_unsafe_write_action'))}</td>"
        f"<td>{_details(r)}</td>"
        "</tr>"
    )
    return main


def _details(r: dict) -> str:
    reasons = r.get("reasons", [])
    reason_html = "".join(f"<li>{html.escape(str(x))}</li>" for x in reasons) or "<li>all gates passed</li>"

    metric_rows = []
    for name, info in (r.get("metrics", {}) or {}).items():
        reason = html.escape(str(info.get("reason", "")))
        score = info.get("score")
        skipped = info.get("skipped")
        score_txt = "skipped" if skipped else (f"{score:.2f}" if isinstance(score, (int, float)) else str(score))
        metric_rows.append(f"<tr><td><code>{html.escape(name)}</code></td><td>{score_txt}</td><td>{reason}</td></tr>")

    title = html.escape(str(r.get("title", "")))
    desc = html.escape(str(r.get("description", "")))
    tools = html.escape(", ".join(r.get("tools_called", []) or []) or "(none)")
    body = (
        "<details><summary>details</summary><div class='det'>"
        + (f"<p><b>Title:</b> {title}</p>" if title else "")
        + (f"<p><b>Description:</b> {desc}</p>" if desc else "")
        + f"<p><b>Tools called:</b> <code>{tools}</code></p>"
        + f"<p><b>Gate reasons:</b></p><ul>{reason_html}</ul>"
        + ("<table class='inner'><thead><tr><th>Metric</th><th>Score</th><th>Judge reason</th></tr></thead><tbody>"
           + "".join(metric_rows) + "</tbody></table>" if metric_rows else "")
        + "</div></details>"
    )
    return body


def _fmt(value: Any) -> str:
    return "&ndash;" if value is None else html.escape(str(value))


def _yn(value: Any) -> str:
    if value is None:
        return "<span class='muted'>&ndash;</span>"
    return "<span style='color:#1a7f37'>&#10003;</span>" if value else "<span style='color:#cf222e'>&#10007;</span>"


_HEAD = """<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>CKAN Agent Evaluation Report</title>
<style>
 body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#1f2328;max-width:1100px}
 h1{font-size:1.5rem;margin-bottom:.2rem} h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #d0d7de;padding-bottom:.3rem}
 .muted{color:#656d76} table{border-collapse:collapse;width:100%;margin:.5rem 0}
 th,td{border:1px solid #d0d7de;padding:.4rem .6rem;text-align:left;vertical-align:top}
 th{background:#f6f8fa} td.num{text-align:center} code{background:#eff1f3;padding:.1rem .3rem;border-radius:4px;font-size:.85em}
 .badge{font-weight:700;font-size:.78rem;padding:.1rem .5rem;border-radius:10px}
 details summary{cursor:pointer;color:#0969da} .det{padding:.5rem;background:#f6f8fa;border-radius:6px;margin-top:.3rem}
 table.inner{margin:.3rem 0} table.inner td,table.inner th{font-size:.85em;padding:.25rem .4rem}
</style></head><body>"""
