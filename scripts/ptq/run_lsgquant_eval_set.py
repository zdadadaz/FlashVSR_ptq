#!/usr/bin/env python3
"""LSGQuant PTQ conversion/evaluation manifest builder.

PR-3 used this entrypoint for A8W8 DRAQ+VOLTS eval. PR-9 extends the
same contract to A4W4 LSGQuant full eval: DRAQ activation QDQ, VOLTS fallback
policy, QAO low-rank residual conversion, and explicit quality/compression
report placeholders. Heavy video rendering remains outside dry-run mode.
Scope is DiT-only; Wan VAE remains unquantized.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.ptq.fakequant_convert import load_lsgquant_layer_policy


def converted_checkpoint_path(out_dir: Path, mode: str, rank: int = 32) -> Path:
    if mode == "a4w4":
        return out_dir / f"dit_lsgquant_{mode}_rank{rank}_qao.safetensors"
    return out_dir / f"dit_lsgquant_{mode}_draq_volts.safetensors"


def build_lsgquant_convert_command(
    checkpoint: Path,
    calibration_cache: Path,
    policy: Path,
    mode: str,
    out_dir: Path,
    enable_bias_correction: bool = False,
    rank: int = 32,
    qao_rounds: int = 4,
    rotation: str = "identity",
) -> list[str]:
    if mode == "a4w4":
        return [
            sys.executable,
            "scripts/ptq/lsgquant_convert.py",
            "--checkpoint",
            str(checkpoint),
            "--calibration_cache",
            str(calibration_cache),
            "--policy",
            str(policy),
            "--mode",
            mode,
            "--rank",
            str(rank),
            "--qao_rounds",
            str(qao_rounds),
            "--rotation",
            rotation,
            "--activation_qdq_mode",
            "draq_symmetric",
            "--output",
            str(converted_checkpoint_path(out_dir, mode, rank=rank)),
        ]

    cmd = [
        sys.executable,
        "scripts/ptq/fakequant_convert.py",
        "--checkpoint",
        str(checkpoint),
        "--calibration_cache",
        str(calibration_cache),
        "--policy",
        str(policy),
        "--mode",
        mode,
        "--activation_qdq_mode",
        "draq_symmetric",
    ]
    if enable_bias_correction:
        cmd.append("--enable_bias_correction")
    cmd.extend([
        "--output",
        str(converted_checkpoint_path(out_dir, mode, rank=rank)),
    ])
    return cmd


def _cache_summary(calibration_cache: Path) -> dict[str, Any]:
    raw = json.loads(calibration_cache.read_text())
    layers = {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}
    return {
        "path": str(calibration_cache),
        "schema_version": raw.get("_metadata", {}).get("schema_version"),
        "layers": len(layers),
        "layers_with_mu_var": sum(1 for v in layers.values() if "mu_var" in v),
        "layers_with_act_mean": sum(1 for v in layers.values() if "act_mean" in v),
        "metadata": raw.get("_metadata", {}),
    }


def _fallback_counts(entries: dict[str, dict[str, Any]], policy_summary: dict[str, Any]) -> dict[str, int]:
    explicit = policy_summary.get("fallback_counts")
    if isinstance(explicit, dict):
        return {str(k): int(v) for k, v in explicit.items()}
    counts = {"a4w4_rank16": 0, "a4w4_rank32_light": 0, "a8w8_rank32": 0, "fp16_skip": 0}
    for entry in entries.values():
        mode = entry.get("mode")
        rank = int(entry.get("rank", 0) or 0)
        adaptation = entry.get("adaptation")
        if mode == "fp16_skip":
            counts["fp16_skip"] += 1
        elif mode == "a8w8":
            counts["a8w8_rank32"] += 1
        elif mode == "a4w4" and rank >= 32 and adaptation == "light":
            counts["a4w4_rank32_light"] += 1
        elif mode == "a4w4":
            counts["a4w4_rank16"] += 1
    return counts


def build_lsgquant_eval_manifest(
    checkpoint: Path,
    calibration_cache: Path,
    policy: Path,
    mode: str,
    out_dir: Path,
    limit: int | None = None,
    enable_bias_correction: bool = False,
    rank: int = 32,
    qao_rounds: int = 4,
    rotation: str = "identity",
) -> dict[str, Any]:
    entries, policy_summary = load_lsgquant_layer_policy(policy)
    converted = converted_checkpoint_path(out_dir, mode, rank=rank)
    command = build_lsgquant_convert_command(
        checkpoint,
        calibration_cache,
        policy,
        mode,
        out_dir,
        enable_bias_correction=enable_bias_correction,
        rank=rank,
        qao_rounds=qao_rounds,
        rotation=rotation,
    )
    is_pr9 = mode == "a4w4"
    manifest = {
        "schema_version": "flashvsr.lsgquant.a4w4_eval_manifest.v1" if is_pr9 else "flashvsr.lsgquant.eval_manifest.v1",
        "paper": "arXiv:2602.03182v1",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "checkpoint": str(checkpoint),
        "converted_checkpoint": str(converted),
        "calibration_cache": _cache_summary(calibration_cache),
        "policy": str(policy),
        "policy_summary": policy_summary,
        "mode": mode,
        "activation_qdq_mode": "draq_symmetric",
        "bias_correction": enable_bias_correction,
        "convert_command": command,
        "eval_limit": limit,
        "quality_gate": {
            "baseline": "static_a8w8" if not is_pr9 else "a8w8_draq_volts_policy",
            "candidate": "a8w8_draq_volts_policy" if not is_pr9 else "a4w4_lsgquant_mixed_policy",
            "metrics": ["psnr", "temporal_drift"],
            "requirement": "candidate should beat static A8W8 or eliminate worst artifact cases before PR4/PR5" if not is_pr9 else "A4W4 mixed policy should render complete videos without NaN/Inf and produce explainable fallback counts",
        },
    }
    if is_pr9:
        fallback_counts = _fallback_counts(entries, policy_summary)
        manifest.update(
            {
                "qao": {"rank": int(rank), "rounds": int(qao_rounds), "rotation": rotation},
                "fallback_counts": fallback_counts,
                "compression_report": {
                    "fakequant_quality_only": True,
                    "deployment_acceleration": "not claimed; TensorRT/int kernels are PR-10 boundary",
                    "fallback_counts": fallback_counts,
                },
                "quality_delta": {
                    "status": "pending_eval",
                    "baseline": "fp16_or_a8w8_draq_volts",
                    "metrics": {"psnr_delta": None, "ssim_delta": None, "lpips_delta": None, "temporal_drift_delta": None},
                },
            }
        )
    else:
        manifest["experimental_options"] = {
            "bias_correction": {
                "enabled": enable_bias_correction,
                "opt_in_flag": "--enable_bias_correction",
                "status": "experimental_opt_in",
                "default": False,
                "rationale": "Disabled by default after PR3 smoke ablation showed PSNR regression.",
            }
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/run LSGQuant DRAQ+VOLTS/QAO mixed-policy eval")
    parser.add_argument("--checkpoint", required=True, type=Path, help="FP16 DiT checkpoint")
    parser.add_argument("--calibration_cache", required=True, type=Path, help="Calibration cache with mu_var")
    parser.add_argument("--policy", required=True, type=Path, help="LSGQuant/VOLTS policy JSON")
    parser.add_argument("--mode", default="a8w8", choices=["a8w8", "a4w4"], help="Candidate mode")
    parser.add_argument("--rank", type=int, default=32, help="PR-9 QAO low-rank branch rank")
    parser.add_argument("--qao_rounds", type=int, default=4, help="PR-9 QAO residual/SVD refinement rounds")
    parser.add_argument("--rotation", default="identity", choices=["identity", "hadamard"], help="Optional PR-9 QAO/input rotation")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for checkpoint + manifest")
    parser.add_argument("--limit", type=int, default=None, help="Eval video limit recorded in manifest")
    parser.add_argument("--dry_run", action="store_true", help="Only write manifest; do not run conversion")
    parser.add_argument("--enable_bias_correction", action="store_true", help="Opt into experimental mean-based bias correction for A8W8; ignored by A4W4 QAO")
    args = parser.parse_args()

    manifest = build_lsgquant_eval_manifest(
        checkpoint=args.checkpoint,
        calibration_cache=args.calibration_cache,
        policy=args.policy,
        mode=args.mode,
        out_dir=args.out_dir,
        limit=args.limit,
        enable_bias_correction=args.enable_bias_correction,
        rank=args.rank,
        qao_rounds=args.qao_rounds,
        rotation=args.rotation,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "lsgquant_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[LSGQuantEval] Manifest → {manifest_path}")
    print(f"[LSGQuantEval] Convert command: {' '.join(manifest['convert_command'])}")

    if not args.dry_run:
        subprocess.run(manifest["convert_command"], check=True)


if __name__ == "__main__":
    main()
