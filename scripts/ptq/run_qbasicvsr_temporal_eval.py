#!/usr/bin/env python3
"""Run/sync QBasicVSR-inspired FlashVSR PTQ eval and leaderboard rows."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.experiments.static_mixed_qat.render_leaderboard import render_html  # noqa: E402
from scripts.experiments.static_mixed_qat.update_leaderboard import build_leaderboard_row, write_jsonl_row  # noqa: E402


def _read_policy(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _mode_counts(policy: dict) -> tuple[int, int, int]:
    counts = policy.get("counts", {})
    a8 = int(counts.get("a8w8", 0) + counts.get("a8w4", 0))
    a16 = int(counts.get("a16w8", 0) + counts.get("a16w4", 0) + counts.get("fp16_skip", 0))
    a4 = int(counts.get("a4w4", 0))
    return a8, a16, a4


def _quantize_mode(policy: dict) -> str:
    qdq = policy.get("activation_qdq_mode", "draq_symmetric")
    if qdq == "draq_symmetric":
        return "FakeQuant_A8W8_DRAQ"
    return "FakeQuant_A8W8"


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_report(path: Path, *, run_id: str, row: dict, policy: dict, manifest: dict, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        f"# QBasicVSR FlashVSR temporal eval report — {run_id}",
        "",
        f"- dry_run: {dry_run}",
        f"- psnr_vs_fp16_mean: {row.get('psnr_vs_fp16_mean')}",
        f"- activation_qdq_mode: {row.get('activation_qdq_mode')}",
        f"- clipping_method: {row.get('clipping')}",
        f"- teacher_ft_steps: {row.get('teacher_ft_steps')}",
        f"- static_ablation_label: {row.get('static_ablation_label')}",
        f"- policy: {row.get('policy')}",
        f"- checkpoint: {row.get('checkpoint')}",
        f"- reproduce_script: {row.get('reproduce_script')}",
        f"- psnr_json: {row.get('psnr_json')}",
        f"- eval_set: {row.get('eval_set')}",
        f"- quant_scope: {policy.get('quant_scope')}",
        f"- wan_vae_quantized: {policy.get('wan_vae_quantized')}",
        f"- base_bits: {policy.get('base_bits')}",
        f"- video_bit_factor: {policy.get('video_bit_factor')}",
        f"- FAB: {policy.get('fab')}",
        f"- flow_backend: {policy.get('flow_backend')}",
        f"- counts: `{json.dumps(policy.get('counts', {}), sort_keys=True)}`",
        "",
        "## Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run QBasicVSR temporal FlashVSR eval and leaderboard update")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_video", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--daily_dir", default="/home/user/SynologyDrive/daily")
    ap.add_argument("--leaderboard", default="outputs/static_mixed_qat/leaderboard.jsonl")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--fp16_video", default="", help="Existing FP16 video to reuse")
    ap.add_argument("--ptq_video", default="", help="Existing PTQ video to reuse")
    ap.add_argument("--clipping_method", default="qbasicvsr_temporal", choices=["none", "qbasicvsr_temporal", "minmax_ema", "omse", "omse_teacher_clipft"])
    ap.add_argument("--teacher_ft_steps", type=int, default=0)
    ap.add_argument("--static_ablation_label", default="")
    ap.add_argument("--reproduce_script", default="", help="Script that reproduces this leaderboard row/eval")
    args = ap.parse_args()

    root = Path.cwd()
    policy = _read_policy(args.policy)
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/qbasicvsr/eval") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fp16_video = Path(args.fp16_video) if args.fp16_video else out_dir / "fp16.mp4"
    ptq_video = Path(args.ptq_video) if args.ptq_video else out_dir / "ptq.mp4"
    psnr_json = out_dir / "psnr.json"
    manifest_path = out_dir / "manifest.json"

    if not args.fp16_video:
        _run([sys.executable, "cli_main.py", "--input", args.input_video, "--output", str(fp16_video), "--scale", "4", "--mode", "tiny", "--end_frame", str(args.frames), "--quantize_mode", "None"], cwd=root, dry_run=args.dry_run)
    if not args.ptq_video:
        _run([sys.executable, "cli_main.py", "--input", args.input_video, "--output", str(ptq_video), "--scale", "4", "--mode", "tiny", "--end_frame", str(args.frames), "--ckpt_path", args.checkpoint, "--quantize_mode", _quantize_mode(policy)], cwd=root, dry_run=args.dry_run)

    if not args.dry_run:
        for video in (fp16_video, ptq_video):
            if not video.exists() or video.stat().st_size == 0:
                raise RuntimeError(f"Missing/non-empty video required for PSNR: {video}")
        _run([sys.executable, "scripts/compare_video_psnr.py", str(fp16_video), str(ptq_video), "--out-json", str(psnr_json)], cwd=root, dry_run=False)
    else:
        # Keep leaderboard parsing deterministic in dry-run tests without
        # pretending this is a measured quality result.
        psnr_json.write_text(json.dumps({"psnr_avg_db": 0.0, "dry_run": True}, indent=2))

    a8, a16, a4 = _mode_counts(policy)
    notes = (
        "QBasicVSR-inspired temporal policy; quant_scope=dit_linear_only; "
        f"wan_vae_quantized=false; bv={policy.get('video_bit_factor')}; base_bits={policy.get('base_bits')}; "
        f"FAB={policy.get('fab')}; flow_backend={policy.get('flow_backend', 'proxy')}; a4_layers={a4}; "
        f"clipping_method={args.clipping_method}; teacher_ft_steps={args.teacher_ft_steps}; "
        f"static_ablation_label={args.static_ablation_label}"
    )
    manifest = {
        "schema_version": "flashvsr.qbasicvsr.temporal_eval.v1",
        "run_id": args.run_id,
        "policy": str(args.policy),
        "checkpoint": str(args.checkpoint),
        "input_video": args.input_video,
        "frames": args.frames,
        "fp16_video": str(fp16_video),
        "ptq_video": str(ptq_video),
        "psnr_json": str(psnr_json),
        "eval_set": args.eval_set,
        "quantize_mode": _quantize_mode(policy),
        "clipping_method": args.clipping_method,
        "teacher_ft_steps": args.teacher_ft_steps,
        "static_ablation_label": args.static_ablation_label,
        "reproduce_script": args.reproduce_script,
        "notes": notes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    row = build_leaderboard_row(
        run_id=args.run_id,
        policy=args.policy,
        checkpoint=args.checkpoint,
        manifest=manifest_path,
        psnr_json=psnr_json,
        reproduce_script=args.reproduce_script or None,
        a8_layers=a8,
        a16_layers=a16,
        activation_qdq_mode=policy.get("activation_qdq_mode", "draq_symmetric"),
        clipping=args.clipping_method,
        eval_set=args.eval_set,
        notes=notes,
        teacher_ft_steps=args.teacher_ft_steps,
        static_ablation_label=args.static_ablation_label,
        fab=policy.get("fab"),
        quant_scope=policy.get("quant_scope"),
        wan_vae_quantized=bool(policy.get("wan_vae_quantized", False)),
    )
    if not args.dry_run and row.get("psnr_vs_fp16_mean") is None:
        raise RuntimeError("Leaderboard row would have blank psnr_vs_fp16_mean")
    write_jsonl_row(args.leaderboard, row)
    html_path = Path(args.leaderboard).with_suffix(".html")
    html_path.write_text(render_html(args.leaderboard))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    daily = Path(args.daily_dir)
    prefix = datetime.now().strftime("%Y%m%d")
    _copy_if_exists(Path(args.leaderboard), daily / f"{prefix}_flashvsr_static_mixed_leaderboard.jsonl")
    _copy_if_exists(html_path, daily / f"{prefix}_flashvsr_static_mixed_leaderboard.html")
    report_path = daily / f"{stamp}_qbasicvsr_flashvsr_report.md"
    write_report(report_path, run_id=args.run_id, row=row, policy=policy, manifest=manifest, dry_run=args.dry_run)
    print(f"[qbasicvsr] leaderboard → {args.leaderboard}")
    print(f"[qbasicvsr] daily report → {report_path}")


if __name__ == "__main__":
    main()
