#!/usr/bin/env python3
"""PR-3 LSGQuant A8W8 DRAQ + VOLTS mixed-policy eval manifest.

This script prepares the conversion/evaluation contract without hiding the heavy
GPU render step. In dry-run mode it writes a manifest and the exact conversion
command; without dry-run it first runs fakequant_convert.py to materialize the
mixed-policy checkpoint. Scope is DiT-only; Wan VAE remains unquantized.
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


def converted_checkpoint_path(out_dir: Path, mode: str) -> Path:
    return out_dir / f"dit_lsgquant_{mode}_draq_volts.safetensors"


def build_lsgquant_convert_command(
    checkpoint: Path,
    calibration_cache: Path,
    policy: Path,
    mode: str,
    out_dir: Path,
    enable_bias_correction: bool = False,
) -> list[str]:
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
        str(converted_checkpoint_path(out_dir, mode)),
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


def build_lsgquant_eval_manifest(
    checkpoint: Path,
    calibration_cache: Path,
    policy: Path,
    mode: str,
    out_dir: Path,
    limit: int | None = None,
    enable_bias_correction: bool = False,
) -> dict[str, Any]:
    _, policy_summary = load_lsgquant_layer_policy(policy)
    converted = converted_checkpoint_path(out_dir, mode)
    command = build_lsgquant_convert_command(checkpoint, calibration_cache, policy, mode, out_dir, enable_bias_correction=enable_bias_correction)
    return {
        "schema_version": "flashvsr.lsgquant.eval_manifest.v1",
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
            "baseline": "static_a8w8",
            "candidate": "a8w8_draq_volts_policy",
            "metrics": ["psnr", "temporal_drift"],
            "requirement": "candidate should beat static A8W8 or eliminate worst artifact cases before PR4/PR5",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/run PR-3 LSGQuant A8W8 DRAQ+VOLTS mixed-policy eval")
    parser.add_argument("--checkpoint", required=True, type=Path, help="FP16 DiT checkpoint")
    parser.add_argument("--calibration_cache", required=True, type=Path, help="PR-2 calibration cache with mu_var")
    parser.add_argument("--policy", required=True, type=Path, help="PR-2 LSGQuant/VOLTS policy JSON")
    parser.add_argument("--mode", default="a8w8", choices=["a8w8"], help="PR-3 candidate mode")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for checkpoint + manifest")
    parser.add_argument("--limit", type=int, default=None, help="Eval video limit recorded in manifest")
    parser.add_argument("--dry_run", action="store_true", help="Only write manifest; do not run conversion")
    parser.add_argument("--enable_bias_correction", action="store_true", help="Opt into experimental mean-based bias correction; disabled by default after PR3 smoke showed PSNR regression")
    args = parser.parse_args()

    manifest = build_lsgquant_eval_manifest(
        checkpoint=args.checkpoint,
        calibration_cache=args.calibration_cache,
        policy=args.policy,
        mode=args.mode,
        out_dir=args.out_dir,
        limit=args.limit,
        enable_bias_correction=args.enable_bias_correction,
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
