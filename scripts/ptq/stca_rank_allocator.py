"""STCA-style rank allocation for LSGQuant low-rank residual layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
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


def _score(stats: dict[str, Any]) -> float:
    vals = _flatten(stats.get("mu_var", stats.get("act_mean", stats.get("act_scale"))))
    if not vals:
        mins = _flatten(stats.get("act_min")); maxs = _flatten(stats.get("act_max"))
        vals = [abs(b - a) for a, b in zip(mins, maxs)] if mins and maxs else []
    if not vals:
        return 0.0
    return float(sum(abs(x) for x in vals) / len(vals))


def build_rank_policy(calibration: dict[str, Any], r_min: int = 16, r_max: int = 64) -> dict[str, Any]:
    scored = {name: _score(stats if isinstance(stats, dict) else {}) for name, stats in calibration.items() if not name.startswith("_")}
    if not scored:
        return {"schema_version": "flashvsr.stca_rank_policy.v1", "layers": {}}
    values = sorted(scored.values())
    lo = values[len(values) // 4]
    hi = values[(3 * len(values)) // 4]
    layers = {}
    for name, score in scored.items():
        rank = r_min
        if score >= hi:
            rank = min(r_max, r_min + 16)
        elif score <= lo:
            rank = r_min
        else:
            rank = min(r_max, r_min + 8)
        rank = max(r_min, min(r_max, int(round(rank / 8) * 8)))
        layers[name] = {"rank": rank, "score": score, "mode": "a4w4", "activation_qdq_mode": "draq_static_s"}
    return {"schema_version": "flashvsr.stca_rank_policy.v1", "r_min": r_min, "r_max": r_max, "thresholds": {"low": lo, "high": hi}, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build STCA-style LSGQuant rank policy")
    parser.add_argument("--calibration_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--r_min", type=int, default=16)
    parser.add_argument("--r_max", type=int, default=64)
    args = parser.parse_args()
    calibration = json.loads(Path(args.calibration_cache).read_text())
    policy = build_rank_policy(calibration, r_min=args.r_min, r_max=args.r_max)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2))
    print(f"[STCA] layers={len(policy['layers'])} -> {out}")


if __name__ == "__main__":
    main()
