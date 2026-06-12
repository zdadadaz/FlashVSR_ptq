"""Build a per-layer Hadamard rotation cache from calibration outlier stats."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _flatten(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def _amax_vector(stats: dict[str, Any]) -> list[float]:
    mins = _flatten(stats.get("act_min"))
    maxs = _flatten(stats.get("act_max"))
    if mins and maxs and len(mins) == len(maxs):
        return [max(abs(a), abs(b)) for a, b in zip(mins, maxs)]
    return [abs(x) for x in _flatten(stats.get("act_scale", stats.get("scale")))]


def inter_channel_variance_score(stats: dict[str, Any]) -> float:
    vals = _amax_vector(stats)
    if len(vals) < 2:
        return 0.0
    mu = abs(mean(vals))
    if mu < 1e-12:
        return 0.0
    return float(pstdev(vals) / mu)


def build_hadamard_cache(calibration: dict[str, Any], variance_threshold: float = 2.0, seed: int = 42) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    enabled = 0
    for name, stats in calibration.items():
        if name.startswith("_"):
            continue
        score = inter_channel_variance_score(stats if isinstance(stats, dict) else {})
        vals = _amax_vector(stats if isinstance(stats, dict) else {})
        in_features = len(vals)
        in_features_padded = 1 << (max(in_features, 1) - 1).bit_length()
        is_enabled = score >= variance_threshold and in_features > 0
        enabled += int(is_enabled)
        layers[name] = {
            "enabled": bool(is_enabled),
            "score": score,
            "seed": seed,
            "in_features": in_features,
            "in_features_padded": in_features_padded,
            "rotation": "hadamard" if is_enabled else "identity",
            "reason": "inter-channel amax variance above threshold" if is_enabled else "low inter-channel amax variance",
        }
    return {
        "schema_version": "flashvsr.hadamard_cache.v1",
        "seed": seed,
        "variance_threshold": variance_threshold,
        "enabled_layers": enabled,
        "total_layers": len(layers),
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Hadamard rotation cache from calibration stats")
    parser.add_argument("--calibration_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variance_threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    calibration = json.loads(Path(args.calibration_cache).read_text())
    cache = build_hadamard_cache(calibration, variance_threshold=args.variance_threshold, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=2))
    print(f"[HadamardCache] enabled_layers={cache['enabled_layers']} total_layers={cache['total_layers']} -> {out}")


if __name__ == "__main__":
    main()
