#!/usr/bin/env python3
"""VOLTS-guided LSGQuant QAT-lite adaptation CLI.

PR8 keeps the tested path CPU/dry-run friendly while defining the manifest and
freeze/adaptation contract used by the heavier DiT fine-tuning runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.quantization.qat import build_volts_adaptation_plan


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_manifest(args: argparse.Namespace, policy: dict[str, Any]) -> dict[str, Any]:
    plan = build_volts_adaptation_plan(policy, light_steps=args.light_steps, full_steps=args.full_steps)
    return {
        "schema_version": "flashvsr.lsgquant.volts_adaptation_manifest.v1",
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": args.student_checkpoint,
        "policy": args.policy,
        "calibration_manifest": args.calibration_manifest,
        "output": args.out,
        "status": "dry_run" if args.dry_run else "pending_heavy_dit_adaptation",
        "adaptation_plan": plan,
        "training": {
            "light_steps": int(args.light_steps),
            "full_steps": int(args.full_steps),
            "lr": float(args.lr),
            "temporal_loss_weight": float(args.temporal_loss_weight),
            "early_stop_loss": args.early_stop_loss,
        },
        "psnr_boundary": "PR8 validates freeze/adaptation contract; run PSNR after full checkpoint adaptation/eval.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashVSR LSGQuant VOLTS-guided QAT-lite adaptation")
    parser.add_argument("--teacher_checkpoint", required=True, help="FP DiT teacher checkpoint")
    parser.add_argument("--student_checkpoint", required=True, help="LSGQuant student checkpoint")
    parser.add_argument("--policy", required=True, help="VOLTS tier policy JSON")
    parser.add_argument("--calibration_manifest", required=True, help="DiT-ready calibration/adaptation sample manifest")
    parser.add_argument("--light_steps", type=int, default=30)
    parser.add_argument("--full_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temporal_loss_weight", type=float, default=0.05)
    parser.add_argument("--early_stop_loss", type=float, default=None)
    parser.add_argument("--out", required=True, help="Adapted checkpoint output path")
    parser.add_argument("--manifest", default="", help="Adaptation manifest output path")
    parser.add_argument("--dry_run", action="store_true", help="Write manifest only; do not load DiT checkpoints")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = _read_json(args.policy)
    manifest_path = args.manifest or str(Path(args.out).with_suffix(Path(args.out).suffix + ".manifest.json"))
    manifest = build_manifest(args, policy)
    _write_json(manifest_path, manifest)
    if args.dry_run:
        print(f"[VOLTS QAT-lite] wrote manifest: {manifest_path}")
        return
    raise NotImplementedError(
        "Full DiT VOLTS adaptation is intentionally not launched by PR8 CLI yet; "
        "use --dry_run to validate policy/freeze contract or wire this manifest "
        "into scripts/qat/finetune_fakequant_dit.py for GPU training."
    )


if __name__ == "__main__":
    main()
