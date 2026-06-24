"""Full LSGQuant conversion entrypoint scaffold for W4A4 low-rank residual PTQ.

The CPU conversion core is implemented in src.models.quantization.qao. This CLI
exposes the master-schedule D flags and supports dry-run manifest generation;
full FlashVSR checkpoint conversion can be enabled by wiring model loading and
state_dict saving around convert_model_to_lsgquant_qao.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rank_policy(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if "layers" in raw:
        return raw["layers"]
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVSR LSGQuant full W4A4 conversion")
    parser.add_argument("--checkpoint", default="", help="FP DiT checkpoint")
    parser.add_argument("--rank_policy", required=False, default="", help="STCA rank policy JSON")
    parser.add_argument("--output", required=False, default="outputs/lsgquant/fakequant_lsgquant_w4a4.safetensors")
    parser.add_argument("--manifest", default="", help="Optional manifest JSON path")
    parser.add_argument("--weight_mode", default="w4", choices=["w4", "w8"])
    parser.add_argument("--activation_qdq_mode", default="draq_static_s")
    parser.add_argument("--calibration_dataset", default="datasets/train/HQ-VSR")
    parser.add_argument("--num_videos", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    policy = load_rank_policy(args.rank_policy) if args.rank_policy else {}
    manifest = {
        "schema_version": "flashvsr.lsgquant_full_convert.v1",
        "checkpoint": args.checkpoint,
        "rank_policy": args.rank_policy or None,
        "output": args.output,
        "weight_mode": args.weight_mode,
        "activation_qdq_mode": args.activation_qdq_mode,
        "calibration_dataset": args.calibration_dataset,
        "num_videos": args.num_videos,
        "num_samples": args.num_samples,
        "default_rank": args.rank,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "policy_layers": len(policy),
        "status": "dry_run_manifest" if args.dry_run else "not_executed_requires_gpu_wiring",
    }
    manifest_path = Path(args.manifest or str(Path(args.output).with_suffix(".manifest.json")))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[LSGQuantFullConvert] manifest -> {manifest_path}")
    if not args.dry_run:
        raise SystemExit("Full checkpoint conversion is not enabled in this scaffold; use --dry_run for contract validation.")


if __name__ == "__main__":
    main()
