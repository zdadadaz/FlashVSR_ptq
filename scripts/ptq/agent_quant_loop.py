#!/usr/bin/env python3
"""Agent-driven outer loop for FlashVSR PTQ experiments.

Concept (from industry practice: NVIDIA kernel loop / CodeMender verifier loop):
the LLM agent NEVER touches the GPU. It proposes configs, this harness executes
the deterministic calibrate -> convert -> render -> PSNR pipeline, and returns a
compact machine-readable verdict. PSNR-vs-FP16 is the oracle; failure logs are
tail-included so the agent can diagnose which stage/layer broke.

Layout produced under --run-dir (default outputs/agent_loop/<run_name>):
  calib_cache/   per-variant calibration JSON (skipped if exists)
  ckpts/         per-variant converted fakequant checkpoints
  videos/        rendered clips (FP16 refs + each variant)
  metrics/       per-clip PSNR JSON + validation_set_psnr_summary.json
  logs/          full stdout/stderr of every subprocess
  reports/<ts>_summary.md   human/agent-readable run report
  manifest.json  the full command plan for this run

Usage:
  python scripts/ptq/agent_quant_loop.py --config configs/agent_quant_loop_example.json --dry-run
  python scripts/ptq/agent_quant_loop.py --config configs/agent_quant_loop_example.json

Exit code: 0 if every variant passes its gate_psnr_db, 1 otherwise (the agent
loop branches on this). Re-runs are incremental: existing caches/videos are
reused, so an agent can tweak one variant and re-run cheaply.

Config schema (see configs/agent_quant_loop_example.json):
  checkpoint      FP16 DiT checkpoint used for calibration + conversion
  clips           list of clip basenames under data_dir (e.g. "bowing_cif")
  data_dir        dir containing <clip>.mp4 low-res inputs
  fp16_ref_dir    dir with <clip>_fp16_first16.mp4 references; auto-rendered
                  with quantize_mode None into this dir if missing
  render          fixed cli_main.py flags shared by all renders
  variants[]      name, quantize_mode, gate_psnr_db, calib{}, convert{}
                  (calib/convert kwargs map 1:1 to fakequant_calibrate.py /
                  fakequant_convert.py CLI flags; null/empty values omitted)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_TAIL_LINES = 40
FPS_RE = re.compile(r"\((\d+\.?\d*)\s+FPS\)")


def extract_fps(log_path: Path) -> float | None:
    """Pull the '(NN.NN FPS)' line out of a cli_main.py render log."""
    try:
        with log_path.open() as f:
            m = FPS_RE.search(f.read())
        return float(m.group(1)) if m else None
    except OSError:
        return None


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()


def usable(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def run_cmd(cmd: list[str], log_path: Path) -> tuple[bool, str]:
    """Run a subprocess, tee output to log_path. Returns (ok, tail-of-log)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    with log_path.open() as f:
        tail = "".join(f.readlines()[-LOG_TAIL_LINES:])
    return proc.returncode == 0, tail


def render_cmd(python: str, render: dict, extra: list[str], inp: Path, out: Path) -> list[str]:
    """Build the cli_main.py render command shared by refs and variants."""
    return [
        python, "cli_main.py",
        "--input", str(inp), "--output", str(out),
        "--model", render.get("model", "FlashVSR-v1.1"),
        "--vae_model", render.get("vae_model", "Wan2.1"),
        "--scale", str(render.get("scale", 4)),
        "--mode", render.get("mode", "full"),
        "--precision", render.get("precision", "fp16"),
        "--device", render.get("device", "cuda:0"),
        "--attention_mode", render.get("attention_mode", "sdpa"),
        "--start_frame", str(render.get("start_frame", 0)),
        "--end_frame", str(render.get("end_frame", 16)),
        "--seed", str(render.get("seed", 0)),
    ] + extra


def build_calib_cmd(python: str, checkpoint: str, calib: dict, cache: Path) -> list[str]:
    cmd = [python, "scripts/ptq/fakequant_calibrate.py",
           "--checkpoint", checkpoint, "--output_cache", str(cache)]
    flag_map = {
        "mode": "--mode", "dataset_train": "--dataset_train", "num_videos": "--num_videos",
        "num_samples": "--num_samples", "calib_frames": "--calib_frames",
        "latent_size": "--latent_size", "seed": "--seed",
        "vae_path": "--vae_path", "vae_model": "--vae_model", "video": "--video",
    }
    for key, flag in flag_map.items():
        val = calib.get(key)
        if val is not None and val != "":
            cmd += [flag, str(val)]
    return cmd


def build_convert_cmd(python: str, checkpoint: str, convert: dict, cache: Path, ckpt_out: Path) -> list[str]:
    cmd = [python, "scripts/ptq/fakequant_convert.py",
           "--checkpoint", checkpoint, "--output", str(ckpt_out)]
    if usable(cache):
        cmd += ["--calibration_cache", str(cache)]
    flag_map = {
        "mode": "--mode", "static_quality_policy": "--static_quality_policy",
        "activation_qdq_mode": "--activation_qdq_mode", "draq_qrange": "--draq_qrange",
        "policy_json": "--policy_json", "smoothquant_cache": "--smoothquant_cache",
        "weight_rounding": "--weight_rounding",
    }
    for key, flag in flag_map.items():
        val = convert.get(key)
        if val is not None and val != "":
            cmd += [flag, str(val)]
    if convert.get("enable_bias_correction"):
        cmd.append("--enable_bias_correction")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="FlashVSR PTQ agent outer loop")
    ap.add_argument("--config", required=True, help="Run config JSON")
    ap.add_argument("--run-dir", default=None, help="Override output run directory")
    ap.add_argument("--python", default=None, help="Python interpreter for subcommands")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the full command plan + manifest, execute nothing")
    ap.add_argument("--metrics-only", action="store_true",
                    help="Skip calib/convert/render; only recompute PSNR metrics from existing videos")
    ap.add_argument("--db", default=os.environ.get("AGENT_RESULT_DB",
                    str(ROOT / "outputs" / "agent_loop" / "results.db")),
                    help="SQLite result database to auto-ingest after the run")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    python = args.python or os.environ.get("AGENT_PYTHON") or str(ROOT / ".venv/bin/python")
    if not usable(Path(python)):
        python = sys.executable

    run_name = cfg.get("run_name") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "outputs" / "agent_loop" / run_name
    videos_dir = run_dir / "videos"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    calib_dir = run_dir / "calib_cache"
    ckpt_dir = run_dir / "ckpts"
    reports_dir = run_dir / "reports"
    for d in (videos_dir, metrics_dir, logs_dir, calib_dir, ckpt_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    clips = cfg["clips"]
    data_dir = ROOT / cfg.get("data_dir", "data/lowres")
    checkpoint = cfg["checkpoint"]
    render = cfg.get("render", {})
    variants = cfg["variants"]

    # FP16 references (the oracle ground truth) — cached across runs.
    fp16_ref_dir = Path(cfg["fp16_ref_dir"]) if cfg.get("fp16_ref_dir") else videos_dir / "fp16_ref"
    fp16_ref_dir.mkdir(parents=True, exist_ok=True)
    ref_paths = {clip: fp16_ref_dir / f"{clip}_fp16_first16.mp4" for clip in clips}

    # ------------------------------------------------------------------
    # Plan: every step the loop will take, as stage dicts.
    # ------------------------------------------------------------------
    plan: list[str] = []
    for clip in clips:
        if not usable(ref_paths[clip]):
            cmd = render_cmd(python, render, ["--quantize_mode", "None"],
                             data_dir / f"{clip}.mp4", ref_paths[clip])
            plan.append(" ".join(cmd))

    variant_plan: dict[str, list[dict]] = {}
    for v in variants:
        vs = slug(v["name"])
        cache = calib_dir / f"{vs}_calib.json"
        ckpt_out = ckpt_dir / f"{vs}.safetensors"
        steps: list[dict] = []
        if not args.metrics_only:
            if not usable(cache):
                steps.append({"stage": "calibrate",
                              "cmd": build_calib_cmd(python, checkpoint, v.get("calib", {}), cache),
                              "log": logs_dir / f"{vs}_calibrate.log"})
            if not usable(ckpt_out):
                steps.append({"stage": "convert",
                              "cmd": build_convert_cmd(python, checkpoint, v.get("convert", {}), cache, ckpt_out),
                              "log": logs_dir / f"{vs}_convert.log"})
            for clip in clips:
                out = videos_dir / f"{clip}_{vs}_first16.mp4"
                if not usable(out):
                    steps.append({"stage": "render", "clip": clip,
                                  "cmd": render_cmd(python, render,
                                                   ["--quantize_mode", v["quantize_mode"],
                                                    "--ckpt_path", str(ckpt_out)],
                                                   data_dir / f"{clip}.mp4", out),
                                  "log": logs_dir / f"{vs}_{slug(clip)}_render.log"})
        for clip in clips:
            out = videos_dir / f"{clip}_{vs}_first16.mp4"
            psnr_json = metrics_dir / f"{clip}_psnr_fp16_vs_{vs}_first16.json"
            steps.append({"stage": "psnr", "clip": clip, "psnr_json": psnr_json,
                          "cmd": [python, "scripts/compare_video_psnr.py",
                                  str(ref_paths[clip]), str(out), "--out-json", str(psnr_json)]})
        variant_plan[v["name"]] = steps
        for s in steps:
            plan.append(" ".join(s["cmd"]))

    manifest = {
        "run_dir": str(run_dir),
        "python": python,
        "checkpoint": checkpoint,
        "clips": clips,
        "fp16_ref_dir": str(fp16_ref_dir),
        "variants": [
            {"name": v["name"], "quantize_mode": v["quantize_mode"],
             "gate_psnr_db": v.get("gate_psnr_db"),
             "steps": [s["stage"] for s in variant_plan[v["name"]]]}
            for v in variants
        ],
        "commands": plan,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "manifest": manifest}, indent=2))
        return 0

    # ------------------------------------------------------------------
    # Execute.
    # ------------------------------------------------------------------
    # Oracle first: ensure FP16 refs exist.
    for clip in clips:
        ref = ref_paths[clip]
        if not usable(ref):
            ok, tail = run_cmd(render_cmd(python, render, ["--quantize_mode", "None"],
                                         data_dir / f"{clip}.mp4", ref),
                              logs_dir / f"fp16_ref_{slug(clip)}.log")
            if not ok:
                print(json.dumps({"error": f"FP16 reference render failed for {clip}",
                                  "log_tail": tail}, indent=2))
                return 1

    all_pass = True
    rows = []
    for v in variants:
        vs = slug(v["name"])
        gate = v.get("gate_psnr_db")
        clip_rows = []
        failed_stage = None
        log_tail = ""
        for step in variant_plan[v["name"]]:
            if step["stage"] == "psnr":
                try:
                    subprocess.run(step["cmd"], cwd=ROOT, check=True,
                                   capture_output=True, text=True)
                    metric = json.loads(step["psnr_json"].read_text())
                    psnrs = metric.get("psnr_per_frame_db", [])
                    std = (sum((p - metric["psnr_avg_db"]) ** 2 for p in psnrs) / len(psnrs)) ** 0.5 if psnrs else None
                    fps = extract_fps(logs_dir / f"{vs}_{slug(step['clip'])}_render.log")
                    clip_rows.append({
                        "clip": step["clip"],
                        "frames": metric["frames"],
                        "psnr_avg_db": metric["psnr_avg_db"],
                        "psnr_min_db": metric["psnr_min_db"],
                        "psnr_std_db": std,
                        "fps": fps,
                    })
                except Exception as e:
                    failed_stage = f"psnr ({step['clip']}): {e}"
                    break
            else:
                ok, tail = run_cmd(step["cmd"], step["log"])
                if not ok:
                    failed_stage = step["stage"]
                    log_tail = tail
                    break
        mean_avg = (sum(r["psnr_avg_db"] for r in clip_rows) / len(clip_rows)) if clip_rows else None
        worst_min = min((r["psnr_min_db"] for r in clip_rows), default=None)
        passed = (failed_stage is None and gate is not None
                  and mean_avg is not None and mean_avg >= gate)
        if not passed:
            all_pass = False
        rows.append({
            "variant": v["name"],
            "quantize_mode": v["quantize_mode"],
            "gate_psnr_db": gate,
            "mean_psnr_avg_db": mean_avg,
            "worst_frame_psnr_db": worst_min,
            "clips": clip_rows,
            "passed": passed,
            "failed_stage": failed_stage,
            "log_tail": log_tail,
        })
        print(json.dumps({k: rows[-1][k] for k in ("variant", "mean_psnr_avg_db",
                                                   "worst_frame_psnr_db", "passed", "failed_stage")},
                         indent=2), flush=True)

    summary = {
        "run": str(run_dir),
        "run_id": run_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "variants": rows,
        "all_passed": all_pass,
    }
    (metrics_dir / "validation_set_psnr_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report (repo convention: dated filename in reports/)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lines = [f"# Agent PTQ loop — {run_name} ({ts})", ""]
    lines.append("| variant | mode | mean PSNR (dB) | worst frame (dB) | gate | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        mean = f"{r['mean_psnr_avg_db']:.2f}" if r["mean_psnr_avg_db"] is not None else "n/a"
        worst = f"{r['worst_frame_psnr_db']:.2f}" if r["worst_frame_psnr_db"] is not None else "n/a"
        verdict = "PASS" if r["passed"] else f"FAIL ({r['failed_stage'] or 'below gate'})"
        lines.append(f"| {r['variant']} | {r['quantize_mode']} | {mean} | {worst} | {r['gate_psnr_db']} | {verdict} |")
    for r in rows:
        if r["failed_stage"] and r["log_tail"]:
            lines += ["", f"## Failure tail: {r['variant']} ({r['failed_stage']})",
                      "```", r["log_tail"].rstrip(), "```"]
    report = reports_dir / f"{ts}_summary.md"
    report.write_text("\n".join(lines) + "\n")

    # Auto-ingest into the cross-run result DB (the "AI Researcher" reads the
    # DB, not raw run folders). Best-effort: never fail the run over the DB.
    db_note = "skipped"
    try:
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        subprocess.run([python, str(ROOT / "scripts" / "ptq" / "agent_result_db.py"),
                        "--db", args.db,
                        "--ingest", str(metrics_dir / "validation_set_psnr_summary.json"),
                        "--config", str(run_dir / "config.json")],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        db_note = args.db
    except Exception as e:
        db_note = f"ingest failed: {e}"

    print(json.dumps({"summary_json": str(metrics_dir / "validation_set_psnr_summary.json"),
                      "report": str(report), "all_passed": all_pass, "db": db_note}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
