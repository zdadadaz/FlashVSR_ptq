#!/usr/bin/env python3
"""QBasicVSR TS-LBA token-space sensitivity collection for FlashVSR DiT Linears.

The lightweight CLI materializes the 306 WanVideoDiT Linear layer names and a
deterministic token-proxy sensitivity smoke cache.  The pure tensor helper is
used by tests and by future real activation hooks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.ptq.fakequant_convert import build_dit, load_checkpoint  # noqa: E402
from src.models.quantization.policy import classify_layer_name  # noqa: E402

SCHEMA_VERSION = "flashvsr.qbasicvsr.temporal_sensitivity.v1"


def compute_spatial_temporal_sensitivity(x: torch.Tensor, frames: int = 1) -> dict:
    xf = x.detach().float()
    if xf.dim() == 3:  # [B,L,C]
        spatial = xf.std(dim=-1, unbiased=False).mean().item()
        b, l, c = xf.shape
        f = max(1, min(int(frames), l))
        usable = (l // f) * f
        if f > 1 and usable > 0:
            xt = xf[:, :usable, :].reshape(b, f, usable // f, c)
            temporal = xt.std(dim=1, unbiased=False).mean().item()
        else:
            temporal = 0.0
        quality = "approx_token"
    elif xf.dim() == 2:  # [B,C]
        spatial = xf.std(dim=-1, unbiased=False).mean().item()
        temporal = 0.0
        quality = "shape_special"
    else:
        spatial = xf.reshape(-1, xf.shape[-1]).std(dim=-1, unbiased=False).mean().item()
        temporal = 0.0
        quality = "fallback_flattened"
    return {"spatial_sensitivity": float(spatial), "temporal_sensitivity": float(temporal), "shape_quality": quality}


def list_dit_linear_layers(checkpoint: str | None = None) -> list[str]:
    model = build_dit()
    if checkpoint:
        model = load_checkpoint(checkpoint, model)
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _deterministic_score(name: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def build_smoke_sensitivity(layer_names: list[str], *, frames: int = 8) -> dict:
    entries = {}
    for name in sorted(layer_names):
        group = classify_layer_name(name)
        # Bias attention layers slightly higher so TS-LBA policies are non-degenerate.
        group_bias = {"self_attn": 0.25, "cross_attn": 0.18, "ffn": 0.10, "embed": 0.35, "time": 0.35, "head": 0.30}.get(group, 0.05)
        spatial = min(1.0, group_bias + 0.75 * _deterministic_score(name, "space"))
        temporal = min(1.0, group_bias + 0.75 * _deterministic_score(name, "temp"))
        entries[name] = {
            "spatial_sensitivity": float(spatial),
            "temporal_sensitivity": float(temporal),
            "activation_shape_samples": [[1, frames * 64, 1536]] if group not in {"embed", "time", "head"} else [[1, 1536]],
            "group": group,
            "shape_quality": "approx_token",
        }
    entries["_metadata"] = {
        "schema_version": SCHEMA_VERSION,
        "num_layers_expected": 306,
        "num_layers_observed": len(layer_names),
        "quant_scope": "dit_linear_only",
        "wan_vae_quantized": False,
        "shape_quality": "approx_token",
        "collector": "deterministic_smoke_proxy",
    }
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect QBasicVSR TS-LBA sensitivities for FlashVSR DiT Linear layers")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, default=8)
    # Compatibility placeholders for the planned GPU collector.
    ap.add_argument("--dataset_train", default="")
    ap.add_argument("--num_videos", type=int, default=2)
    ap.add_argument("--calib_frames", type=int, default=8)
    ap.add_argument("--num_samples", type=int, default=8)
    args = ap.parse_args()
    layers = list_dit_linear_layers(args.checkpoint or None)
    report = build_smoke_sensitivity(layers, frames=args.frames or args.calib_frames)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[qbasicvsr] temporal sensitivity layers={len(layers)} → {out}")


if __name__ == "__main__":
    main()
