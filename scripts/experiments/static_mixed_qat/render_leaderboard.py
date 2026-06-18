#!/usr/bin/env python3
"""Render FlashVSR static mixed QAT leaderboard.jsonl to a standalone HTML file."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda row: (row.get("psnr_vs_fp16_mean") is None, -(row.get("psnr_vs_fp16_mean") or -1e9), row.get("a16_layers") or 10**9))
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def render_html(jsonl_path: str | Path) -> str:
    rows = load_rows(jsonl_path)
    columns = [
        "run_id", "psnr_vs_fp16_mean", "a8_layers", "a16_layers", "activation_qdq_mode",
        "clipping", "bias_correction", "qat", "observer", "freeze_step", "total_steps",
        "eval_set", "policy", "checkpoint", "reproduce_script", "notes",
    ]
    body = []
    for row in rows:
        tds = []
        for col in columns:
            value = row.get(col)
            if col in {"policy", "checkpoint", "reproduce_script"} and value:
                text = html.escape(Path(str(value)).name)
                tds.append(f'<td title="{html.escape(str(value))}">{text}</td>')
            else:
                tds.append(f"<td>{_cell(value)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FlashVSR Static Mixed QAT Leaderboard</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; background: #0b1020; color: #e8ecff; }}
h1 {{ margin-bottom: 0.25rem; }}
.meta {{ color: #aab3d6; margin-bottom: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #253052; padding: 6px 8px; vertical-align: top; }}
th {{ background: #18213d; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #101831; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 5px; background: #273762; }}
</style>
</head>
<body>
<h1>FlashVSR Static Mixed QAT Leaderboard</h1>
<div class="meta">Source: <code>{html.escape(str(jsonl_path))}</code>. Metrics marked PSNR-vs-FP16 are consistency metrics unless GT columns are populated.</div>
<table>
<thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FlashVSR static mixed leaderboard HTML")
    parser.add_argument("--leaderboard", default="outputs/static_mixed_qat/leaderboard.jsonl")
    parser.add_argument("--output", default="outputs/static_mixed_qat/leaderboard.html")
    args = parser.parse_args()
    html_text = render_html(args.leaderboard)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text)
    print(f"[leaderboard] rendered {out}")


if __name__ == "__main__":
    main()
