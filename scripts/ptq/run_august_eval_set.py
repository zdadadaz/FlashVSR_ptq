#!/usr/bin/env python3
"""Run August PTQ recovery FP16-vs-mixed validation set."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = os.environ.get("RUN_ID") or Path("/tmp/flashvsr_august_eval_run_id").read_text().strip()
CKPT = ROOT / "outputs" / "ptq_recovery" / "2026-08-personA" / "eval" / RUN_ID / "checkpoints" / "dit_august_mixed_recovery_v1_actmean_biascorr.safetensors"
BASE = ROOT / "outputs" / "ptq_recovery" / "2026-08-personA" / "eval" / RUN_ID
VIDEOS = BASE / "videos" / "validation_set"
METRICS = BASE / "metrics"
REPORTS = BASE / "reports"

CLIPS = [
    "bowing_cif",
    "bus_cif",
    "carphone_qcif",
    "city_cif",
    "coastguard_cif",
]


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}; log={log_path}")


def main() -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    python = str(ROOT / ".venv/bin/python")
    rows = []
    for clip in CLIPS:
        inp = ROOT / "data" / "lowres" / f"{clip}.mp4"
        if not inp.exists():
            raise FileNotFoundError(inp)
        fp16 = VIDEOS / f"{clip}_fp16_first16.mp4"
        mixed = VIDEOS / f"{clip}_august_mixed_biascorr_first16.mp4"
        common = [
            python, "cli_main.py",
            "--input", str(inp),
            "--model", "FlashVSR-v1.1",
            "--vae_model", "Wan2.1",
            "--scale", "4",
            "--mode", "full",
            "--precision", "fp16",
            "--device", "cuda:0",
            "--attention_mode", "sdpa",
            "--start_frame", "0",
            "--end_frame", "16",
            "--seed", "0",
        ]
        if not fp16.exists() or fp16.stat().st_size == 0:
            run(common + ["--output", str(fp16), "--quantize_mode", "None"], REPORTS / f"{clip}_fp16.log")
        if not mixed.exists() or mixed.stat().st_size == 0:
            run(common + ["--output", str(mixed), "--quantize_mode", "FakeQuant_A8W8", "--ckpt_path", str(CKPT)], REPORTS / f"{clip}_mixed.log")
        psnr_json = METRICS / f"{clip}_psnr_fp16_vs_august_mixed_biascorr_first16.json"
        out = subprocess.check_output([python, "scripts/compare_video_psnr.py", str(fp16), str(mixed), "--out-json", str(psnr_json)], cwd=ROOT, text=True)
        metric = json.loads(psnr_json.read_text())
        rows.append({
            "clip": clip,
            "frames": metric["frames"],
            "psnr_avg_db": metric["psnr_avg_db"],
            "psnr_min_db": metric["psnr_min_db"],
            "psnr_max_db": metric["psnr_max_db"],
            "fp16_video": str(fp16.relative_to(ROOT)),
            "mixed_video": str(mixed.relative_to(ROOT)),
            "metric_json": str(psnr_json.relative_to(ROOT)),
        })
        print(out)
    avg = sum(r["psnr_avg_db"] for r in rows) / len(rows)
    summary = {
        "run_id": RUN_ID,
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "clips": rows,
        "mean_psnr_avg_db": avg,
        "min_clip_avg_psnr_db": min(r["psnr_avg_db"] for r in rows),
        "max_clip_avg_psnr_db": max(r["psnr_avg_db"] for r in rows),
    }
    out_path = METRICS / "validation_set_psnr_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
