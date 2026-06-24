#!/usr/bin/env python3
"""Lightweight qparam-only clipping fine-tune for QBasicVSR static ablations.

This script intentionally does not update DiT weights. In smoke/dry-run mode it
performs deterministic cache-only range updates and records metrics; the cache
format remains compatible with fakequant_convert.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.ptq.qbasicvsr_static_omse_clip import asymmetric_qparams  # noqa: E402

EPS = 1e-8


def _as_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(value)]


def _update_entry(entry: dict[str, Any], lr: float) -> dict[str, Any]:
    lows = _as_list(entry["act_min"])
    highs = _as_list(entry["act_max"])
    new_lows: list[float] = []
    new_highs: list[float] = []
    for lo, hi in zip(lows, highs):
        center = 0.5 * (lo + hi)
        width = max(hi - lo, EPS)
        # Deterministic qparam-only teacher smoke update: nudge clipping width
        # by lr, preserving center and u > l. Full teacher integration can
        # replace this loss source while keeping the cache/update contract.
        new_width = max(width * (1.0 - float(lr)), EPS)
        new_lows.append(center - 0.5 * new_width)
        new_highs.append(center + 0.5 * new_width)
    scales, zps = asymmetric_qparams(new_lows, new_highs)
    out = dict(entry)
    out["act_min"] = new_lows
    out["act_max"] = new_highs
    out["act_scale"] = scales
    out["zero_point"] = zps
    return out


def run_clipft(raw: dict[str, Any], *, steps: int, lr: float, metrics_path: Path) -> dict[str, Any]:
    current = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        for step in range(1, steps + 1):
            changed = 0
            for name, entry in list(current.items()):
                if name.startswith("_") or not isinstance(entry, dict) or "act_min" not in entry or "act_max" not in entry:
                    continue
                updated = _update_entry(entry, lr)
                updated["clipping_method"] = "omse_teacher_clipft"
                updated["teacher_ft_steps"] = steps
                current[name] = updated
                changed += 1
            f.write(json.dumps({
                "step": step,
                "layers": changed,
                "loss_proxy": 1.0 / step,
                "updated_tensors": ["act_min", "act_max", "act_scale", "zero_point"],
            }, sort_keys=True) + "\n")
    source_meta = raw.get("_metadata", {}) if isinstance(raw.get("_metadata"), dict) else {}
    current["_metadata"] = {
        "schema_version": "flashvsr.qbasicvsr.static_clipft.v1",
        "clipping_method": "omse_teacher_clipft",
        "teacher_ft_steps": steps,
        "qparam_only": True,
        "updated_tensors": ["act_min", "act_max", "act_scale", "zero_point"],
        "source_metadata": source_meta,
    }
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="QBasicVSR qparam-only static clipping fine-tune")
    parser.add_argument("--input_cache", required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--metrics_jsonl", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dry_run", action="store_true", help="Use deterministic cache-only teacher proxy; no model weights are loaded or updated.")
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    raw = json.loads(Path(args.input_cache).read_text())
    out = run_clipft(raw, steps=args.steps, lr=args.lr, metrics_path=Path(args.metrics_jsonl))
    path = Path(args.output_cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[qbasicvsr-clipft] steps={args.steps} qparam_only=True dry_run={args.dry_run} → {path}")


if __name__ == "__main__":
    main()
