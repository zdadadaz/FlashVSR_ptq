#!/usr/bin/env python3
"""Evaluate August-B static-A8/A16W8 SmoothQuant+AdaRound PTQ baseline vs FP16."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "static_ptq_ablation" / "augustB_policy_sq_adaround_staticA8_actmean"
CKPT = RUN / "dit_augustB_staticA8_a16w8_sq_adaround_biascorr.safetensors"
VIDEOS = RUN / "videos" / "validation_set"
METRICS = RUN / "metrics"
REPORTS = RUN / "reports"
FP16_SOURCE = ROOT / "outputs" / "ptq_recovery" / "2026-08-personA" / "eval" / "20260602_173721_august_B_train10_static_asym" / "videos" / "validation_set"
CLIPS = ["bowing_cif", "bus_cif", "carphone_qcif", "city_cif", "coastguard_cif"]


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}; log={log_path}")


def main() -> None:
    if not CKPT.exists() or CKPT.stat().st_size == 0:
        raise FileNotFoundError(CKPT)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    python = str(ROOT / ".venv/bin/python")
    rows = []
    for clip in CLIPS:
        inp = ROOT / "data" / "lowres" / f"{clip}.mp4"
        fp16 = FP16_SOURCE / f"{clip}_fp16_first16.mp4"
        if not fp16.exists() or fp16.stat().st_size == 0:
            # Fallback to generating into this run if the reusable August-B reference is absent.
            fp16 = VIDEOS / f"{clip}_fp16_first16.mp4"
            common_fp = [
                python, "cli_main.py", "--input", str(inp), "--model", "FlashVSR-v1.1",
                "--vae_model", "Wan2.1", "--scale", "4", "--mode", "full",
                "--precision", "fp16", "--device", "cuda:0", "--attention_mode", "sdpa",
                "--start_frame", "0", "--end_frame", "16", "--seed", "0",
            ]
            if not fp16.exists() or fp16.stat().st_size == 0:
                run(common_fp + ["--output", str(fp16), "--quantize_mode", "None"], REPORTS / f"{clip}_fp16.log")
        mixed = VIDEOS / f"{clip}_augustB_staticA8_a16w8_sq_adaround_biascorr_first16.mp4"
        common = [
            python, "cli_main.py", "--input", str(inp), "--model", "FlashVSR-v1.1",
            "--vae_model", "Wan2.1", "--scale", "4", "--mode", "full",
            "--precision", "fp16", "--device", "cuda:0", "--attention_mode", "sdpa",
            "--start_frame", "0", "--end_frame", "16", "--seed", "0",
        ]
        if not mixed.exists() or mixed.stat().st_size == 0:
            run(common + ["--output", str(mixed), "--quantize_mode", "FakeQuant_A8W8", "--ckpt_path", str(CKPT)], REPORTS / f"{clip}_mixed.log")
        psnr_json = METRICS / f"{clip}_psnr_fp16_vs_augustB_staticA8_sq_first16.json"
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
        print(out, flush=True)
    summary = {
        "run": str(RUN.relative_to(ROOT)),
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "clips": rows,
        "mean_psnr_avg_db": sum(r["psnr_avg_db"] for r in rows) / len(rows),
        "min_clip_avg_psnr_db": min(r["psnr_avg_db"] for r in rows),
        "max_clip_avg_psnr_db": max(r["psnr_avg_db"] for r in rows),
    }
    out_path = METRICS / "validation_set_psnr_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
