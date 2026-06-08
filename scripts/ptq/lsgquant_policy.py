#!/usr/bin/env python3
"""Generate LSGQuant/VOLTS layer policy from FakeQuant calibration cache."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.quantization.policy import build_lsgquant_volts_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PR-2 LSGQuant/VOLTS layer policy from calibration cache")
    parser.add_argument("--calibration_cache", required=True, help="Input calibration JSON with per-layer mu_var")
    parser.add_argument("--out", required=True, help="Output policy JSON path")
    parser.add_argument("--delta1", type=float, default=0.001, help="Frozen/light threshold or percentile cut")
    parser.add_argument("--delta2", type=float, default=0.075, help="Light/full threshold or percentile cut")
    parser.add_argument(
        "--threshold_mode",
        choices=["absolute", "percentile"],
        default="absolute",
        help="Interpret delta1/delta2 as absolute mu_var thresholds or percentile cuts",
    )
    parser.add_argument(
        "--default_mode",
        choices=["a16w8", "a8w8", "a16w4", "a8w4", "fp16_skip"],
        default="a8w8",
        help="Layer quantization mode to emit for all DiT Linear layers",
    )
    parser.add_argument(
        "--activation_qdq_mode",
        choices=["static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric", "draq_symmetric"],
        default="draq_symmetric",
        help="Activation QDQ mode for A8 default_mode policies",
    )
    args = parser.parse_args()

    with open(args.calibration_cache) as f:
        calibration_cache = json.load(f)

    policy = build_lsgquant_volts_policy(
        calibration_cache,
        delta1=args.delta1,
        delta2=args.delta2,
        threshold_mode=args.threshold_mode,
        default_mode=args.default_mode,
        default_activation_qdq_mode=args.activation_qdq_mode,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(policy, f, indent=2)

    print(f"[LSGQuantPolicy] wrote {args.out}")
    print(f"[LSGQuantPolicy] counts={policy['counts']} thresholds={policy['thresholds']}")


if __name__ == "__main__":
    main()
