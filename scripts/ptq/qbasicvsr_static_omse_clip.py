#!/usr/bin/env python3
"""Build QBasicVSR static-asymmetric OMSE clipping caches.

The output cache is compatible with scripts/ptq/fakequant_convert.py: each layer
keeps act_scale and zero_point, with act_min/act_max replaced by the selected
clipping bounds. If raw act_samples are present, selection uses true simulated
asymmetric int8 QDQ MSE; otherwise it falls back to a deterministic min/max proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

QMIN = -128.0
QMAX = 127.0
EPS = 1e-8


def _as_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(value)]


def _broadcast(values: list[float], n: int) -> list[float]:
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    raise ValueError(f"Cannot broadcast {len(values)} values to {n}")


def asymmetric_qparams(lows: list[float], highs: list[float]) -> tuple[list[float], list[int]]:
    scales: list[float] = []
    zps: list[int] = []
    for lo, hi in zip(lows, highs):
        hi = max(float(hi), float(lo) + EPS)
        scale = max((hi - lo) / (QMAX - QMIN), EPS)
        zp = round(QMIN - lo / scale)
        zp = int(max(QMIN, min(QMAX, zp)))
        scales.append(scale)
        zps.append(zp)
    return scales, zps


def _qdq_mse(samples: list[float], lo: float, hi: float) -> float:
    scale, zp = asymmetric_qparams([lo], [hi])
    scale = scale[0]
    zp = zp[0]
    err = 0.0
    for x in samples:
        raw_x = float(x)
        x_clamped = min(max(raw_x, lo), hi)
        q = round(x_clamped / scale + zp)
        q = max(QMIN, min(QMAX, q))
        x_hat = (q - zp) * scale
        sample_err = (raw_x - x_hat) ** 2
        err += sample_err
    return err / max(1, len(samples))


def _proxy_mse(lo: float, hi: float, base_lo: float, base_hi: float) -> float:
    scale = max((hi - lo) / (QMAX - QMIN), EPS)
    quant_mse = scale * scale / 12.0
    tail = max(0.0, base_lo - lo) ** 2 + max(0.0, hi - base_hi) ** 2
    # Prefer narrower ranges only when they materially reduce quantization step.
    return quant_mse + tail * 1e-5


def select_omse_bounds(entry: dict[str, Any], factors: list[float]) -> dict[str, Any]:
    lows0 = _as_list(entry["act_min"])
    highs0 = _as_list(entry["act_max"])
    n = max(len(lows0), len(highs0))
    lows0 = _broadcast(lows0, n)
    highs0 = _broadcast(highs0, n)
    samples = entry.get("act_samples")
    if samples is not None and samples and not isinstance(samples[0], list):
        samples = [samples]
    best_lows: list[float] = []
    best_highs: list[float] = []
    best_factors: list[float] = []
    best_mses: list[float] = []
    for i, (base_lo, base_hi) in enumerate(zip(lows0, highs0)):
        center = 0.5 * (base_lo + base_hi)
        width = max(base_hi - base_lo, EPS)
        layer_samples = _broadcast([float(x) for x in samples[min(i, len(samples) - 1)]] if samples else [], 0) if False else None
        if samples:
            raw_samples = samples[min(i, len(samples) - 1)]
            layer_samples = [float(x) for x in raw_samples]
        best = None
        for factor in factors:
            factor = float(factor)
            if layer_samples is not None and abs(base_hi) >= abs(base_lo):
                lo = base_lo
                hi = base_lo + width * factor
            elif layer_samples is not None:
                hi = base_hi
                lo = base_hi - width * factor
            else:
                lo = center - 0.5 * width * factor
                hi = center + 0.5 * width * factor
            mse = _qdq_mse(layer_samples, lo, hi) if layer_samples is not None else _proxy_mse(lo, hi, base_lo, base_hi)
            # Bias toward clipping in exact sample mode when error is tied/near-tied.
            tie_break = factor * 1e-12
            key = (mse + tie_break, factor)
            if best is None or key < best[0]:
                best = (key, lo, hi, factor, mse)
        assert best is not None
        _, lo, hi, factor, mse = best
        best_lows.append(lo)
        best_highs.append(hi)
        best_factors.append(float(factor))
        best_mses.append(float(mse))
    scales, zps = asymmetric_qparams(best_lows, best_highs)
    out = {k: v for k, v in entry.items() if k not in {"act_samples", "mu_samples", "mu_samples_mean"}}
    out["act_min"] = best_lows
    out["act_max"] = best_highs
    out["act_scale"] = scales
    out["zero_point"] = zps
    out["clipping_method"] = "omse"
    out["omse_best_factor"] = best_factors
    out["omse_mse"] = best_mses
    return out


def build_omse_cache(raw: dict[str, Any], factors: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    layers = 0
    source_meta = raw.get("_metadata", {})
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        if "act_min" not in entry or "act_max" not in entry:
            out[name] = entry
            continue
        out[name] = select_omse_bounds(entry, factors)
        layers += 1
    out["_metadata"] = {
        "schema_version": "flashvsr.qbasicvsr.static_omse_clip.v1",
        "clipping_method": "omse",
        "layers": layers,
        "factors": factors,
        "source_metadata": source_meta,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QBasicVSR static asymmetric OMSE clipping cache")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factors", default="1.0,0.999,0.995,0.99,0.98,0.95,0.90,0.85,0.80")
    args = parser.parse_args()
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    raw = json.loads(Path(args.input).read_text())
    out = build_omse_cache(raw, factors)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[qbasicvsr-omse] layers={out['_metadata']['layers']} → {path}")


if __name__ == "__main__":
    main()
