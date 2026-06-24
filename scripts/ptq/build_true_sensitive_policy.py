"""Build a mixed-precision policy from true sensitivity measurements.

Top-K% sensitive layers (by MSE) → fp16_skip (keep FP16).
Others → dynamic A8W8 (the mode that already works in your pipeline).

Key design choice: use **dynamic_asymmetric** for non-sensitive layers, NOT
static_asymmetric. Static A8 is the broken mode; this script's whole point is
to recover PSNR by only applying A8 to layers that can tolerate it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.quantization.policy import classify_layer_name


def load_sensitivity_cache(path: str) -> Dict[str, Dict[str, float]]:
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        out[k] = {"mse": float(v["mse"]), "rel_l1": float(v["rel_l1"])}
    return out


def build_true_sensitive_policy(
    sensitivity: Dict[str, Dict[str, float]],
    fp16_skip_ratio: float = 0.20,
    robust_mode: str = "a8w8",
    robust_activation_qdq_mode: str = "dynamic_asymmetric",
) -> Dict[str, Any]:
    """Top-K% sensitive layers → fp16_skip. Others → dynamic A8W8.

    Args:
        sensitivity: {layer_name: {mse, rel_l1}}
        fp16_skip_ratio: fraction of layers to fallback to FP16
        robust_mode: mode for non-sensitive layers (default a8w8)
        robust_activation_qdq_mode: qdq mode for robust layers (default dynamic_asymmetric)
    """
    if not 0.0 < fp16_skip_ratio < 1.0:
        raise ValueError(f"fp16_skip_ratio must be in (0, 1), got {fp16_skip_ratio}")

    sorted_layers = sorted(
        sensitivity.items(),
        key=lambda x: x[1]["mse"],
        reverse=True,
    )
    fp16_skip_count = int(math.ceil(len(sorted_layers) * fp16_skip_ratio))
    fp16_skip_names = {name for name, _ in sorted_layers[:fp16_skip_count]}

    # Per-group breakdown for diagnostics
    group_fp16_skip = Counter()
    group_total = Counter()

    layers: Dict[str, Dict[str, str]] = {}
    for name, info in sensitivity.items():
        group = classify_layer_name(name)
        group_total[group] += 1
        if name in fp16_skip_names:
            layers[name] = {
                "mode": "fp16_skip",
                "group": group,
                "reason": f"top-{fp16_skip_ratio:.0%} true sensitivity MSE={info['mse']:.4e}",
            }
            group_fp16_skip[group] += 1
        else:
            layers[name] = {
                "mode": robust_mode,
                "activation_qdq_mode": robust_activation_qdq_mode,
                "group": group,
                "reason": f"passes true sensitivity threshold MSE={info['mse']:.4e}",
            }

    # Diagnostics
    print(f"\n[PolicyBuilder] Built policy: {fp16_skip_count} fp16_skip / {len(sensitivity)} total")
    print(f"[PolicyBuilder] Per-group fp16_skip ratio:")
    for group in sorted(group_total):
        skip = group_fp16_skip[group]
        total = group_total[group]
        ratio = skip / max(total, 1)
        print(f"  {group:15s}: {skip:3d}/{total:3d} ({ratio:.0%})")

    return {
        "schema_version": "flashvsr.true_sensitivity_policy.v1",
        "name": f"true_sensitivity_top{int(fp16_skip_ratio*100)}pct_fp16skip",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "default": {
            "mode": robust_mode,
            "activation_qdq_mode": robust_activation_qdq_mode,
        },
        "fp16_skip_ratio": fp16_skip_ratio,
        "fp16_skip_count": fp16_skip_count,
        "counts": {
            "fp16_skip": fp16_skip_count,
            robust_mode: len(sensitivity) - fp16_skip_count,
        },
        "group_breakdown": {
            "total": dict(group_total),
            "fp16_skip": dict(group_fp16_skip),
        },
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(description="Build true-sensitivity mixed policy")
    parser.add_argument("--sensitivity", type=str, required=True, help="Input sensitivity JSON")
    parser.add_argument("--output", type=str, required=True, help="Output policy JSON")
    parser.add_argument("--fp16_skip_ratio", type=float, default=0.20, help="Fraction of layers to FP16 (0.0-1.0)")
    parser.add_argument("--robust_mode", type=str, default="a8w8")
    parser.add_argument("--robust_activation_qdq_mode", type=str, default="dynamic_asymmetric",
                        choices=["dynamic_asymmetric", "dynamic_symmetric", "static_asymmetric", "draq_symmetric"])
    args = parser.parse_args()

    sensitivity = load_sensitivity_cache(args.sensitivity)
    print(f"[PolicyBuilder] Loaded {len(sensitivity)} layers from {args.sensitivity}")

    policy = build_true_sensitive_policy(
        sensitivity,
        fp16_skip_ratio=args.fp16_skip_ratio,
        robust_mode=args.robust_mode,
        robust_activation_qdq_mode=args.robust_activation_qdq_mode,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(policy, f, indent=2)
    print(f"\n[PolicyBuilder] Saved policy → {args.output}")


if __name__ == "__main__":
    main()
