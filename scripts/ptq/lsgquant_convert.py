#!/usr/bin/env python3
"""LSGQuant PR6 QAO conversion scaffolding.

This script exposes the conversion contract for producing LSGQuant `(WR, L1, L2)`
checkpoints.  The heavy full-DiT conversion is intentionally conservative: PR6
adds CPU QAO helpers and a manifest/debug CLI with `--max_layers`; future PRs can
wire full checkpoint materialization once the model-side policy is finalized.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def build_lsgquant_qao_manifest(
    checkpoint: Path,
    calibration_cache: Path,
    policy: Path,
    mode: str,
    output: Path,
    rank: int,
    qao_rounds: int,
    max_layers: int | None,
    layer_errors: list[dict[str, Any]] | None = None,
    rotation: str = "identity",
) -> dict[str, Any]:
    """Build the PR6 conversion manifest schema."""

    layers = layer_errors or []
    improved = sum(1 for layer in layers if layer.get("error_after", float("inf")) < layer.get("error_before", float("-inf")))
    return {
        "schema_version": "flashvsr.lsgquant.qao_manifest.v1",
        "paper": "arXiv:2602.03182v1",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "checkpoint": str(checkpoint),
        "calibration_cache": str(calibration_cache) if calibration_cache else "",
        "policy": str(policy) if policy else "",
        "mode": mode,
        "output": str(output),
        "qao": {"rank": int(rank), "rounds": int(qao_rounds), "rotation": rotation},
        "max_layers": max_layers,
        "created_at_unix": time.time(),
        "layers": layers,
        "summary": {
            "layers": len(layers),
            "improved_layers": improved,
            "mean_error_before": (sum(float(x["error_before"]) for x in layers) / len(layers)) if layers else None,
            "mean_error_after": (sum(float(x["error_after"]) for x in layers) / len(layers)) if layers else None,
        },
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FlashVSR DiT checkpoint to LSGQuant QAO low-rank residual format")
    parser.add_argument("--checkpoint", required=True, type=Path, help="FP16 DiT checkpoint")
    parser.add_argument("--calibration_cache", default=Path(""), type=Path, help="Calibration cache JSON")
    parser.add_argument("--policy", default=Path(""), type=Path, help="LSGQuant/VOLTS policy JSON")
    parser.add_argument("--mode", default="a8w8", choices=["a8w8", "a16w8", "a8w4", "a16w4"], help="Quantization mode")
    parser.add_argument("--rank", type=int, default=32, help="QAO low-rank branch rank")
    parser.add_argument("--qao_rounds", type=int, default=4, help="QAO residual/SVD refinement rounds")
    parser.add_argument("--rotation", default="identity", choices=["identity"], help="Rotation path; PR6 supports identity only")
    parser.add_argument("--activation_qdq_mode", default="static_asymmetric", choices=["static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric", "draq_symmetric"], help="Activation QDQ mode for A8 modes")
    parser.add_argument("--draq_qrange", default="signed_symmetric", choices=["signed_symmetric", "signed_full"], help="DRAQ signed int8 clamp range")
    parser.add_argument("--max_layers", type=int, default=None, help="Debug limit for layer-by-layer conversion")
    parser.add_argument("--output", required=True, type=Path, help="Output checkpoint path")
    parser.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON path")
    parser.add_argument("--dry_run", action="store_true", help="Write manifest only; do not materialize checkpoint")
    args = parser.parse_args()

    manifest = build_lsgquant_qao_manifest(
        checkpoint=args.checkpoint,
        calibration_cache=args.calibration_cache,
        policy=args.policy,
        mode=args.mode,
        output=args.output,
        rank=args.rank,
        qao_rounds=args.qao_rounds,
        max_layers=args.max_layers,
        layer_errors=[],
        rotation=args.rotation,
    )
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")

    if args.dry_run:
        _write_json(manifest_path, manifest)
        print(f"[LSGQuant QAO] wrote manifest: {manifest_path}")
        return

    from scripts.ptq.fakequant_convert import build_dit, load_calibration_cache, load_checkpoint, load_lsgquant_layer_policy
    from src.models.quantization.qao import convert_model_to_lsgquant_qao

    print(f"[LSGQuant QAO] loading checkpoint: {args.checkpoint}")
    model = build_dit()
    model = load_checkpoint(str(args.checkpoint), model)
    model.eval()

    act_stats = {}
    if args.calibration_cache and str(args.calibration_cache):
        act_stats = load_calibration_cache(str(args.calibration_cache), device="cpu")

    layer_policy = None
    if args.policy and str(args.policy):
        layer_policy, _ = load_lsgquant_layer_policy(args.policy)

    print(f"[LSGQuant QAO] converting mode={args.mode}, rank={args.rank}, rounds={args.qao_rounds}, max_layers={args.max_layers}")
    layer_errors = convert_model_to_lsgquant_qao(
        model,
        mode=args.mode,
        rank=args.rank,
        rounds=args.qao_rounds,
        rotation=args.rotation,
        act_stats=act_stats,
        layer_policy=layer_policy,
        activation_qdq_mode=args.activation_qdq_mode,
        draq_qrange=args.draq_qrange,
        max_layers=args.max_layers,
    )
    manifest = build_lsgquant_qao_manifest(
        checkpoint=args.checkpoint,
        calibration_cache=args.calibration_cache,
        policy=args.policy,
        mode=args.mode,
        output=args.output,
        rank=args.rank,
        qao_rounds=args.qao_rounds,
        max_layers=args.max_layers,
        layer_errors=layer_errors,
        rotation=args.rotation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".safetensors":
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(args.output))
    else:
        import torch
        torch.save(model.state_dict(), args.output)
    _write_json(manifest_path, manifest)
    print(f"[LSGQuant QAO] saved checkpoint: {args.output}")
    print(f"[LSGQuant QAO] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
