#!/usr/bin/env python3
"""Build a FlashVSR static-mixed NPU handoff manifest/package.

The package is intentionally metadata-first: checkpoints/videos are large and stay in
place, while this script writes a reproducible handoff JSON/Markdown with paths,
checksums, leaderboard rows, and GT-readiness status.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.static_mixed_qat.update_leaderboard import sha256_file
from scripts.qat.prepare_video_manifest import discover_paired_lq_gt


def load_leaderboard_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def select_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("schema_version") == "flashvsr.static_mixed_leaderboard_row.v1"
        and (row.get("checkpoint") or row.get("qat"))
    ]


def best_by_psnr(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("psnr_vs_fp16_mean") is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: float(row["psnr_vs_fp16_mean"]))


def summarize_gt(gt_dir: str | Path | None) -> dict[str, Any]:
    if not gt_dir:
        return {
            "status": "not_provided",
            "paired_samples": 0,
            "message": "No --gt_dir provided; GT PSNR drop gate was not run.",
        }
    root = Path(gt_dir)
    if not root.exists():
        return {
            "status": "missing_path",
            "gt_dir": str(root),
            "paired_samples": 0,
            "message": "Provided --gt_dir does not exist.",
        }
    pairs = discover_paired_lq_gt(root)
    return {
        "status": "ready" if pairs else "no_pairs_found",
        "gt_dir": str(root),
        "paired_samples": len(pairs),
        "pairs_preview": [{"name": p["name"], "lq": str(p["lq"]), "gt": str(p["gt"])} for p in pairs[:10]],
        "message": "Paired LQ/GT dataset discovered." if pairs else "No common LQ/GT, LR/HR, or LQ-Video/GT-Video paired layout found.",
    }


def artifact_entry(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "sha256": sha256_file(p),
        "size_bytes": p.stat().st_size if p.exists() else None,
    }


def infer_training_dataset(row: dict[str, Any]) -> str | None:
    """Infer the training dataset label from explicit fields or notes."""

    for key in ("training_dataset", "train_dataset"):
        if row.get(key):
            return str(row[key])
    text = " ".join(str(row.get(k, "")) for k in ("run_id", "eval_set", "notes", "manifest"))
    if "REDS30" in text or "reds30" in text.lower():
        return "REDS30"
    if "HQ-VSR" in text or "hq-vsr" in text.lower():
        return "HQ-VSR"
    return None


def sync_leaderboard_to_daily(
    leaderboard: str | Path,
    leaderboard_html: str | Path | None,
    daily_dir: str | Path,
    prefix: str = "20260618",
) -> dict[str, str | None]:
    """Copy leaderboard JSONL/HTML into the user's daily folder with stable names."""

    daily = Path(daily_dir)
    daily.mkdir(parents=True, exist_ok=True)
    src_jsonl = Path(leaderboard)
    dst_jsonl = daily / f"{prefix}_flashvsr_static_mixed_leaderboard.jsonl"
    shutil.copy2(src_jsonl, dst_jsonl)
    copied: dict[str, str | None] = {"jsonl": str(dst_jsonl), "html": None}
    if leaderboard_html and Path(leaderboard_html).exists():
        dst_html = daily / f"{prefix}_flashvsr_static_mixed_leaderboard.html"
        shutil.copy2(leaderboard_html, dst_html)
        copied["html"] = str(dst_html)
    return copied


def build_handoff_manifest(*, leaderboard: str | Path, leaderboard_html: str | Path | None, gt_dir: str | Path | None) -> dict[str, Any]:
    rows = load_leaderboard_rows(leaderboard)
    candidates = select_candidate_rows(rows)
    best = best_by_psnr(candidates)
    gt = summarize_gt(gt_dir)
    return {
        "schema_version": "flashvsr.static_mixed_npu_handoff.v1",
        "scope": "DiT Linear only; Wan VAE remains unquantized",
        "npu_constraint": "static activation qparams only",
        "leaderboard": artifact_entry(leaderboard),
        "leaderboard_html": artifact_entry(leaderboard_html),
        "rows_total": len(rows),
        "candidate_rows": len(candidates),
        "best_vs_fp16_run_id": best.get("run_id") if best else None,
        "best_vs_fp16_psnr_db": float(best["psnr_vs_fp16_mean"]) if best else None,
        "gt_readiness": gt,
        "recommended_next_gate": "Run GT-backed PSNR drop once paired LQ/GT path is supplied; target drop <= 0.3-0.4 dB.",
        "artifacts": [
            {
                "run_id": row.get("run_id"),
                "checkpoint": artifact_entry(row.get("checkpoint")),
                "policy": artifact_entry(row.get("policy")),
                "reproduce_script": artifact_entry(row.get("reproduce_script")),
                "manifest": artifact_entry(row.get("manifest")),
                "a8_layers": row.get("a8_layers"),
                "a16_layers": row.get("a16_layers"),
                "qat": row.get("qat"),
                "observer": row.get("observer"),
                "activation_qdq_mode": row.get("activation_qdq_mode"),
                "clipping": row.get("clipping"),
                "psnr_vs_fp16_mean": row.get("psnr_vs_fp16_mean"),
                "eval_set": row.get("eval_set"),
                "training_dataset": infer_training_dataset(row),
                "notes": row.get("notes"),
            }
            for row in candidates
        ],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# FlashVSR Static Mixed NPU Handoff",
        "",
        f"Scope: {manifest['scope']}",
        f"NPU constraint: {manifest['npu_constraint']}",
        "",
        "## Leaderboard",
        "",
        f"Rows total: {manifest['rows_total']}",
        f"Candidate rows: {manifest['candidate_rows']}",
        f"Best FP16-consistency run: {manifest['best_vs_fp16_run_id']} ({manifest['best_vs_fp16_psnr_db']} dB)",
        "",
        "## GT readiness",
        "",
        f"Status: {manifest['gt_readiness']['status']}",
        f"Paired samples: {manifest['gt_readiness']['paired_samples']}",
        f"Message: {manifest['gt_readiness']['message']}",
        "",
        "## Artifact rows",
        "",
    ]
    for row in manifest["artifacts"]:
        lines.extend([
            f"- {row['run_id']}",
            f"  - checkpoint: {row['checkpoint']['path'] if row['checkpoint'] else None}",
            f"  - policy: {row['policy']['path'] if row['policy'] else None}",
            f"  - reproduce: {row['reproduce_script']['path'] if row['reproduce_script'] else None}",
            f"  - A8/A16: {row['a8_layers']}/{row['a16_layers']}",
            f"  - qat/observer: {row['qat']}/{row['observer']}",
            f"  - training_dataset: {row.get('training_dataset')}",
            f"  - qdq/clipping: {row['activation_qdq_mode']}/{row['clipping']}",
            f"  - psnr_vs_fp16_mean: {row['psnr_vs_fp16_mean']}",
        ])
    lines.extend(["", "## Next gate", "", manifest["recommended_next_gate"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FlashVSR static mixed NPU handoff package")
    parser.add_argument("--leaderboard", default="outputs/static_mixed_qat/leaderboard.jsonl")
    parser.add_argument("--leaderboard_html", default="outputs/static_mixed_qat/leaderboard.html")
    parser.add_argument("--gt_dir", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sync_daily_dir", default="", help="Optional daily folder to copy leaderboard JSONL/HTML into")
    parser.add_argument("--daily_prefix", default="20260618")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_handoff_manifest(
        leaderboard=args.leaderboard,
        leaderboard_html=args.leaderboard_html,
        gt_dir=args.gt_dir or None,
    )
    (out_dir / "npu_handoff_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "NPU_HANDOFF.md").write_text(render_markdown(manifest))
    if args.leaderboard_html and Path(args.leaderboard_html).exists():
        shutil.copy2(args.leaderboard_html, out_dir / "leaderboard.html")
    daily_sync = None
    if args.sync_daily_dir:
        daily_sync = sync_leaderboard_to_daily(args.leaderboard, args.leaderboard_html, args.sync_daily_dir, prefix=args.daily_prefix)
        (out_dir / "daily_sync.json").write_text(json.dumps(daily_sync, indent=2))
    print(json.dumps({"out_dir": str(out_dir), "candidate_rows": manifest["candidate_rows"], "gt_status": manifest["gt_readiness"]["status"], "daily_sync": daily_sync}, indent=2))


if __name__ == "__main__":
    main()
