"""Layer policy helpers for FlashVSR FakeQuant PTQ recovery.

The policy format is deliberately small and JSON-serializable so PTQ recovery
experiments can be reproduced from a manifest without changing model code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_LAYER_MODES = {"a16w8", "a8w8", "a16w4", "a8w4", "a4w4", "fp16_skip"}
VALID_ACTIVATION_QDQ_MODES = {
    "static_asymmetric",
    "dynamic_symmetric",
    "dynamic_asymmetric",
    "draq_symmetric",
}


@dataclass(frozen=True)
class LayerDecision:
    mode: str
    activation_qdq_mode: str | None = None
    reason: str = ""


def classify_layer_name(name: str) -> str:
    """Classify a WanVideoDiT Linear layer name into a coarse sensitivity group."""

    if name.startswith("text_embedding"):
        return "embed"
    if name.startswith("time_embedding") or name.startswith("time_projection"):
        return "time"
    if name.startswith("head"):
        return "head"
    if ".ffn." in name:
        return "ffn"
    if ".self_attn." in name:
        return "self_attn"
    if ".cross_attn." in name:
        return "cross_attn"
    return "other"


def _percentile(values: list[float], percentile: float) -> float:
    """Small dependency-free percentile helper for CLI policy generation."""

    if not values:
        raise ValueError("Cannot compute percentile of an empty value list")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def build_lsgquant_volts_policy(
    calibration_cache: dict[str, Any],
    delta1: float = 0.001,
    delta2: float = 0.075,
    threshold_mode: str = "absolute",
    default_mode: str = "a8w8",
    default_activation_qdq_mode: str = "draq_symmetric",
) -> dict[str, Any]:
    """Build an LSGQuant/VOLTS-style three-tier policy from calibration mu_var.

    Tiers:
    - frozen: mu_var <= delta1
    - light:  delta1 < mu_var <= delta2
    - full:   mu_var > delta2

    In percentile mode, delta1/delta2 are percentile cut points over the observed
    mu_var distribution. This avoids degenerate all-frozen/all-full policies when
    FlashVSR's activation distribution differs from the paper defaults.
    """

    if default_mode not in VALID_LAYER_MODES:
        raise ValueError(f"Unsupported default_mode: {default_mode}")
    if default_activation_qdq_mode not in VALID_ACTIVATION_QDQ_MODES:
        raise ValueError(f"Unsupported activation_qdq_mode: {default_activation_qdq_mode}")
    if threshold_mode not in {"absolute", "percentile"}:
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")

    layer_mu_vars: dict[str, float] = {}
    for name, entry in calibration_cache.items():
        if name == "_metadata":
            continue
        if not isinstance(entry, dict) or "mu_var" not in entry:
            raise ValueError(f"Layer {name} is missing required mu_var")
        layer_mu_vars[name] = float(entry["mu_var"])
    if not layer_mu_vars:
        raise ValueError("Calibration cache contains no layer mu_var entries")

    if threshold_mode == "percentile":
        values = list(layer_mu_vars.values())
        resolved_delta1 = _percentile(values, delta1)
        resolved_delta2 = _percentile(values, delta2)
    else:
        resolved_delta1 = float(delta1)
        resolved_delta2 = float(delta2)
    if resolved_delta1 > resolved_delta2:
        raise ValueError("delta1 must be <= delta2 after threshold resolution")

    layers: dict[str, dict[str, Any]] = {}
    counts = {"frozen": 0, "light": 0, "full": 0}
    for name, mu_var in sorted(layer_mu_vars.items()):
        if mu_var <= resolved_delta1:
            tier = "frozen"
        elif mu_var <= resolved_delta2:
            tier = "light"
        else:
            tier = "full"
        counts[tier] += 1
        layers[name] = {
            "mode": default_mode,
            "activation_qdq_mode": default_activation_qdq_mode if default_mode.startswith(("a8", "a4")) else None,
            "tier": tier,
            "mu_var": mu_var,
        }
        if layers[name]["activation_qdq_mode"] is None:
            del layers[name]["activation_qdq_mode"]

    return {
        "schema_version": "flashvsr.lsgquant.policy.v1",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "default": {"mode": default_mode, "activation_qdq_mode": default_activation_qdq_mode},
        "thresholds": {
            "mode": threshold_mode,
            "delta1": float(delta1),
            "delta2": float(delta2),
            "resolved_delta1": resolved_delta1,
            "resolved_delta2": resolved_delta2,
        },
        "tiers": {"frozen": 0, "light": 30, "full": -1},
        "counts": counts,
        "layers": layers,
    }


def build_lsgquant_a4w4_fallback_policy(
    calibration_cache: dict[str, Any],
    delta1: float = 0.001,
    delta2: float = 0.075,
    threshold_mode: str = "absolute",
    fp16_topk: int = 0,
    default_activation_qdq_mode: str = "draq_symmetric",
) -> dict[str, Any]:
    """Build PR9 A4W4 mixed fallback policy from VOLTS ``mu_var`` stats.

    Policy contract follows the PR9 plan:
    - low sensitivity: A4W4 LSGQuant rank-16
    - mid sensitivity: A4W4 LSGQuant rank-32 + light adaptation marker
    - high sensitivity: A8W8 LSGQuant rank-32 fallback
    - optional catastrophic top-k: FP16 skip fallback
    """

    if default_activation_qdq_mode not in VALID_ACTIVATION_QDQ_MODES:
        raise ValueError(f"Unsupported activation_qdq_mode: {default_activation_qdq_mode}")
    if fp16_topk < 0:
        raise ValueError(f"fp16_topk must be >= 0, got {fp16_topk}")

    base = build_lsgquant_volts_policy(
        calibration_cache,
        delta1=delta1,
        delta2=delta2,
        threshold_mode=threshold_mode,
        default_mode="a4w4",
        default_activation_qdq_mode=default_activation_qdq_mode,
    )
    ranked = sorted(
        ((name, float(entry["mu_var"])) for name, entry in base["layers"].items()),
        key=lambda item: item[1],
        reverse=True,
    )
    fp16_names = {name for name, _ in ranked[:fp16_topk]}

    layers: dict[str, dict[str, Any]] = {}
    fallback_counts = {"a4w4_rank16": 0, "a4w4_rank32_light": 0, "a8w8_rank32": 0, "fp16_skip": 0}
    for name, entry in base["layers"].items():
        tier = entry["tier"]
        mu_var = float(entry["mu_var"])
        if name in fp16_names:
            layer = {
                "mode": "fp16_skip",
                "tier": tier,
                "mu_var": mu_var,
                "rank": 0,
                "reason": "catastrophic top-k sensitivity fallback; keep FP16 Linear",
            }
            fallback_counts["fp16_skip"] += 1
        elif tier == "frozen":
            layer = {
                "mode": "a4w4",
                "activation_qdq_mode": default_activation_qdq_mode,
                "tier": tier,
                "mu_var": mu_var,
                "rank": 16,
                "adaptation": "none",
                "reason": "low VOLTS sensitivity: A4W4 LSGQuant rank-16",
            }
            fallback_counts["a4w4_rank16"] += 1
        elif tier == "light":
            layer = {
                "mode": "a4w4",
                "activation_qdq_mode": default_activation_qdq_mode,
                "tier": tier,
                "mu_var": mu_var,
                "rank": 32,
                "adaptation": "light",
                "reason": "mid VOLTS sensitivity: A4W4 LSGQuant rank-32 plus light adaptation",
            }
            fallback_counts["a4w4_rank32_light"] += 1
        else:
            layer = {
                "mode": "a8w8",
                "activation_qdq_mode": default_activation_qdq_mode,
                "tier": tier,
                "mu_var": mu_var,
                "rank": 32,
                "adaptation": "optional_full",
                "reason": "high VOLTS sensitivity: fallback to A8W8 LSGQuant rank-32 before FP16 skip",
            }
            fallback_counts["a8w8_rank32"] += 1
        layers[name] = layer

    return {
        "schema_version": "flashvsr.lsgquant.a4w4_policy.v1",
        "paper": "arXiv:2602.03182v1",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "default": {"mode": "a4w4", "activation_qdq_mode": default_activation_qdq_mode, "rank": 16},
        "thresholds": base["thresholds"],
        "tiers": base["tiers"],
        "counts": base["counts"],
        "fallback_counts": fallback_counts,
        "fp16_topk": int(fp16_topk),
        "layers": layers,
    }


def build_august_mixed_policy(
    layer_names: list[str],
    sensitive_mode: str = "a16w8",
    robust_mode: str = "a8w8",
    robust_activation_qdq_mode: str = "dynamic_asymmetric",
) -> dict[str, Any]:
    """Build Person A August mixed precision policy.

    Sensitive embedding/time/FFN layers keep A16 activations with W8 weights;
    robust attention/head layers use A8W8. This is model-side recovery only and
    keeps Wan VAE out of scope.
    """

    if sensitive_mode not in VALID_LAYER_MODES:
        raise ValueError(f"Unsupported sensitive_mode: {sensitive_mode}")
    if robust_mode not in VALID_LAYER_MODES:
        raise ValueError(f"Unsupported robust_mode: {robust_mode}")
    if robust_activation_qdq_mode not in VALID_ACTIVATION_QDQ_MODES:
        raise ValueError(f"Unsupported activation_qdq_mode: {robust_activation_qdq_mode}")

    sensitive_groups = {"embed", "time", "ffn", "other"}
    layers: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for name in layer_names:
        group = classify_layer_name(name)
        if group in sensitive_groups:
            decision = LayerDecision(
                mode=sensitive_mode,
                reason=f"{group} is activation-sensitive; keep A16 activations for PTQ recovery",
            )
        else:
            decision = LayerDecision(
                mode=robust_mode,
                activation_qdq_mode=robust_activation_qdq_mode if robust_mode.startswith(("a8", "a4")) else None,
                reason=f"{group} is assigned to INT8 activation path",
            )
        layer_entry = {"mode": decision.mode, "reason": decision.reason, "group": group}
        if decision.activation_qdq_mode:
            layer_entry["activation_qdq_mode"] = decision.activation_qdq_mode
        layers[name] = layer_entry
        counts[decision.mode] = counts.get(decision.mode, 0) + 1

    return {
        "schema_version": "flashvsr.fakequant.layer_policy.v1",
        "name": "august_mixed_recovery_v1",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "default": {"mode": robust_mode, "activation_qdq_mode": robust_activation_qdq_mode},
        "counts": counts,
        "layers": layers,
    }


def load_layer_policy(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    layers = raw.get("layers", raw)
    if not isinstance(layers, dict):
        raise ValueError("Layer policy must contain a dict under 'layers'")
    for name, entry in layers.items():
        mode = entry.get("mode") if isinstance(entry, dict) else entry
        if mode not in VALID_LAYER_MODES:
            raise ValueError(f"Layer {name} has unsupported mode: {mode}")
        qdq = entry.get("activation_qdq_mode") if isinstance(entry, dict) else None
        if qdq is not None and qdq not in VALID_ACTIVATION_QDQ_MODES:
            raise ValueError(f"Layer {name} has unsupported activation_qdq_mode: {qdq}")
    return raw


def layer_policy_entries(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    layers = policy.get("layers", policy)
    out: dict[str, dict[str, Any]] = {}
    for name, entry in layers.items():
        out[name] = {"mode": entry} if isinstance(entry, str) else dict(entry)
    return out
