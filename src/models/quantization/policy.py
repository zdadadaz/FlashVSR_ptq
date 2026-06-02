"""Layer policy helpers for FlashVSR FakeQuant PTQ recovery.

The policy format is deliberately small and JSON-serializable so PTQ recovery
experiments can be reproduced from a manifest without changing model code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_LAYER_MODES = {"a16w8", "a8w8", "a16w4", "a8w4", "fp16_skip"}
VALID_ACTIVATION_QDQ_MODES = {
    "static_asymmetric",
    "dynamic_symmetric",
    "dynamic_asymmetric",
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
                activation_qdq_mode=robust_activation_qdq_mode if robust_mode.startswith("a8") else None,
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
