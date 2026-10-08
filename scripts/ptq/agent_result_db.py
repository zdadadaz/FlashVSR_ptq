#!/usr/bin/env python3
"""SQLite result database for FlashVSR PTQ agent loops.

The cross-run history is what turns a one-off sweep into a research loop:
the "AI Researcher" step reads this DB (not raw run folders) to pick the
next hypothesis. Stores per-run configs, per-variant gates, and per-clip
PSNR + FPS so the PSNR-vs-FPS Pareto frontier can be queried directly.

Usage:
  python scripts/ptq/agent_result_db.py --init
  python scripts/ptq/agent_result_db.py --ingest <validation_set_psnr_summary.json> [--config <run_config.json>]
  python scripts/ptq/agent_result_db.py --top 10
  python scripts/ptq/agent_result_db.py --runs
  python scripts/ptq/agent_result_db.py --detail <run_id>
  python scripts/ptq/agent_result_db.py --query quantize_mode=FakeQuant_A8W8

Default DB path: outputs/agent_loop/results.db (override with --db or $AGENT_RESULT_DB).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "outputs" / "agent_loop" / "results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id      TEXT PRIMARY KEY,
  run_dir     TEXT,
  created_at  TEXT,
  config_json TEXT,
  notes       TEXT
);
CREATE TABLE IF NOT EXISTS variants (
  run_id              TEXT NOT NULL,
  variant             TEXT NOT NULL,
  quantize_mode       TEXT,
  gate_psnr_db        REAL,
  mean_psnr_avg_db    REAL,
  worst_frame_psnr_db REAL,
  passed              INTEGER,
  failed_stage        TEXT,
  PRIMARY KEY (run_id, variant)
);
CREATE TABLE IF NOT EXISTS clips (
  run_id        TEXT NOT NULL,
  variant       TEXT NOT NULL,
  clip          TEXT NOT NULL,
  frames        INTEGER,
  psnr_avg_db   REAL,
  psnr_min_db   REAL,
  psnr_std_db   REAL,
  fps           REAL,
  PRIMARY KEY (run_id, variant, clip)
);
"""


def connect(db: str) -> sqlite3.Connection:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def run_id_from(summary: Path) -> str:
    return summary.parent.parent.name if summary.parent.name == "metrics" else summary.stem


def db_path(con: sqlite3.Connection) -> str:
    return con.execute("PRAGMA database_list").fetchone()[2]


def cmd_init(con: sqlite3.Connection) -> None:
    print(json.dumps({"db": db_path(con), "status": "ok"}))


def cmd_ingest(con: sqlite3.Connection, summary_path: Path, config_path: str | None) -> None:
    data = json.loads(summary_path.read_text())
    run_id = data.get("run_id") or run_id_from(summary_path)
    config_json = None
    if config_path:
        config_json = Path(config_path).read_text()
    elif (summary_path.parent.parent / "config.json").exists():
        config_json = (summary_path.parent.parent / "config.json").read_text()
    con.execute(
        "INSERT OR REPLACE INTO runs (run_id, run_dir, created_at, config_json, notes) VALUES (?,?,?,?,?)",
        (run_id, data.get("run"), data.get("timestamp"), config_json, data.get("notes")),
    )
    nv = 0
    for v in data.get("variants", []):
        con.execute(
            "INSERT OR REPLACE INTO variants (run_id, variant, quantize_mode, gate_psnr_db,"
            " mean_psnr_avg_db, worst_frame_psnr_db, passed, failed_stage) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, v["variant"], v.get("quantize_mode"), v.get("gate_psnr_db"),
             v.get("mean_psnr_avg_db"), v.get("worst_frame_psnr_db"),
             int(bool(v.get("passed"))), v.get("failed_stage")),
        )
        nv += 1
        for c in v.get("clips", []):
            con.execute(
                "INSERT OR REPLACE INTO clips (run_id, variant, clip, frames, psnr_avg_db,"
                " psnr_min_db, psnr_std_db, fps) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, v["variant"], c["clip"], c.get("frames"), c.get("psnr_avg_db"),
                 c.get("psnr_min_db"), c.get("psnr_std_db"), c.get("fps")),
            )
    con.commit()
    print(json.dumps({"run_id": run_id, "variants_ingested": nv, "db": db_path(con)}))


def _rows_to_json(rows) -> list[dict]:
    return [dict(r) for r in rows]


def cmd_top(con: sqlite3.Connection, n: int) -> None:
    rows = con.execute(
        """SELECT v.run_id, v.variant, v.quantize_mode, v.mean_psnr_avg_db,
                  v.worst_frame_psnr_db, v.passed,
                  AVG(c.fps) AS avg_fps
           FROM variants v LEFT JOIN clips c ON c.run_id = v.run_id AND c.variant = v.variant
           GROUP BY v.run_id, v.variant
           ORDER BY v.mean_psnr_avg_db DESC NULLS LAST LIMIT ?""",
        (n,),
    ).fetchall()
    print(json.dumps(_rows_to_json(rows), indent=2))


def cmd_runs(con: sqlite3.Connection) -> None:
    rows = con.execute(
        "SELECT r.run_id, r.created_at, COUNT(v.variant) AS n_variants,"
        " SUM(v.passed) AS n_passed FROM runs r LEFT JOIN variants v ON v.run_id = r.run_id"
        " GROUP BY r.run_id ORDER BY r.created_at DESC"
    ).fetchall()
    print(json.dumps(_rows_to_json(rows), indent=2))


def cmd_detail(con: sqlite3.Connection, run_id: str) -> None:
    run = con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    variants = con.execute("SELECT * FROM variants WHERE run_id = ?", (run_id,)).fetchall()
    clips = con.execute("SELECT * FROM clips WHERE run_id = ? ORDER BY variant, clip", (run_id,)).fetchall()
    print(json.dumps({"run": dict(run) if run else None,
                      "variants": _rows_to_json(variants),
                      "clips": _rows_to_json(clips)}, indent=2))


def cmd_query(con: sqlite3.Connection, filters: list[str]) -> None:
    where, params = [], []
    for f in filters:
        if "=" not in f:
            sys.exit(f"bad filter {f!r}, expected col=value")
        col, val = f.split("=", 1)
        col = col.strip()
        if col not in ("quantize_mode", "variant", "run_id", "passed"):
            sys.exit(f"unsupported filter column {col!r}")
        where.append(f"v.{col} = ?")
        params.append(int(val) if col == "passed" else val)
    sql = ("SELECT v.run_id, v.variant, v.quantize_mode, v.mean_psnr_avg_db,"
           " v.worst_frame_psnr_db, v.passed FROM variants v")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY v.mean_psnr_avg_db DESC NULLS LAST"
    print(json.dumps(_rows_to_json(con.execute(sql, params).fetchall()), indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="FlashVSR PTQ result database")
    ap.add_argument("--db", default=os.environ.get("AGENT_RESULT_DB", str(DEFAULT_DB)))
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--ingest", help="validation_set_psnr_summary.json to import")
    ap.add_argument("--config", help="Run config JSON to archive with the run")
    ap.add_argument("--top", type=int, help="Rank variants by mean PSNR")
    ap.add_argument("--runs", action="store_true", help="List runs")
    ap.add_argument("--detail", help="Dump one run (variants + clips)")
    ap.add_argument("--query", action="append", default=[], help="col=value filter (repeatable)")
    args = ap.parse_args()

    con = connect(args.db)
    did_something = False
    if args.init:
        cmd_init(con); did_something = True
    if args.ingest:
        cmd_ingest(con, Path(args.ingest), args.config); did_something = True
    if args.top is not None:
        cmd_top(con, args.top); did_something = True
    if args.runs:
        cmd_runs(con); did_something = True
    if args.detail:
        cmd_detail(con, args.detail); did_something = True
    if args.query:
        cmd_query(con, args.query); did_something = True
    if not did_something:
        ap.print_help()
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
