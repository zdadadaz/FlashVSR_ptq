#!/usr/bin/env python3
"""Build QBasicVSR-inspired temporal mixed-bit policy JSONs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.quantization.policy import build_qbasicvsr_temporal_policy  # noqa: E402


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def decide_layer_bit_factors(sensitivity: dict[str, Any], *, p_space: float = 30.0, p_temp: float = 30.0) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, float]]:
    entries = {k: v for k, v in sensitivity.items() if not k.startswith("_")}
    spatial = {k: float(v.get("spatial_sensitivity", 0.0)) for k, v in entries.items()}
    temporal = {k: float(v.get("temporal_sensitivity", 0.0)) for k, v in entries.items()}
    s_low = _percentile(list(spatial.values()), p_space)
    s_high = _percentile(list(spatial.values()), 100.0 - p_space)
    t_low = _percentile(list(temporal.values()), p_temp)
    t_high = _percentile(list(temporal.values()), 100.0 - p_temp)
    b = {}
    for name in entries:
        if spatial[name] >= s_high and temporal[name] >= t_high:
            b[name] = 1
        elif spatial[name] <= s_low and temporal[name] <= t_low:
            b[name] = -1
        else:
            b[name] = 0
    return b, spatial, temporal, {"p_space": p_space, "p_temp": p_temp, "space_low": s_low, "space_high": s_high, "temp_low": t_low, "temp_high": t_high}


def video_bit_factor_for_path(video_complexity: dict[str, Any], video_path: str | None) -> int:
    if video_path:
        target = str(video_path)
        for entry in video_complexity.get("videos", []):
            if entry.get("path") == target or Path(str(entry.get("path"))).name == Path(target).name:
                return int(entry.get("video_bit_factor", 0))
    videos = video_complexity.get("videos", [])
    return int(videos[0].get("video_bit_factor", 0)) if videos else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build QBasicVSR temporal mixed-bit policy")
    ap.add_argument("--temporal_sensitivity", required=True)
    ap.add_argument("--video_complexity", required=True)
    ap.add_argument("--video_path", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_bits", type=int, default=4)
    ap.add_argument("--p_space", type=float, default=30.0)
    ap.add_argument("--p_temp", type=float, default=30.0)
    ap.add_argument("--activation_qdq_mode", default="draq_symmetric")
    ap.add_argument("--protect_groups", default="embed,time,head")
    ap.add_argument("--force_video_bit_factor", type=int, choices=[-1, 0, 1], default=None)
    ap.add_argument("--disable_a4w4", action="store_true")
    args = ap.parse_args()

    sens = _load_json(args.temporal_sensitivity)
    comp = _load_json(args.video_complexity)
    b_layer, spatial, temporal, thresholds = decide_layer_bit_factors(sens, p_space=args.p_space, p_temp=args.p_temp)
    bv = args.force_video_bit_factor if args.force_video_bit_factor is not None else video_bit_factor_for_path(comp, args.video_path or None)
    thresholds.update({"video": comp.get("thresholds", {}), "flow_backend": comp.get("flow_backend", "proxy")})
    policy = build_qbasicvsr_temporal_policy(
        list(b_layer),
        b_base=args.base_bits,
        video_bit_factor=bv,
        layer_bit_factors=b_layer,
        spatial_sensitivity=spatial,
        temporal_sensitivity=temporal,
        thresholds=thresholds,
        activation_qdq_mode=args.activation_qdq_mode,
        protect_groups=[x for x in args.protect_groups.split(",") if x],
        a4w4_enabled=not args.disable_a4w4,
        flow_backend=comp.get("flow_backend", "proxy"),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2, sort_keys=True))
    print(f"[qbasicvsr] policy layers={len(policy['layers'])} fab={policy['fab']:.4f} counts={policy['counts']} → {out}")


if __name__ == "__main__":
    main()
