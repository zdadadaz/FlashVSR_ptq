#!/usr/bin/env python3
"""Generate PR4 sensitivity-guided FlashVSR DiT FakeQuant layer policies.

The script is intentionally CPU-testable: policy construction is split into pure
helpers, and the CLI can consume a pre-exported layer-name JSON without loading
the full FlashVSR model. When `--checkpoint` is used, only WanVideoDiT
`nn.Linear` layer names are enumerated; Wan VAE remains out of scope.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.quantization.policy import VALID_ACTIVATION_QDQ_MODES, VALID_LAYER_MODES

SENSITIVE_NAME_PATTERNS = [
    "ffn",
    "time_embedding",
    "text_embedding",
    "time_projection",
    "head",
    "LQ_proj_in",
]


def classify_layer_name(name: str) -> tuple[bool, str]:
    """Return whether a layer name should default to sensitive precision."""

    lower = name.lower()
    if "ffn" in lower:
        return True, "ffn layers are sensitive in FlashVSR DiT PTQ"
    if "time_embedding" in lower or "text_embedding" in lower or "time_projection" in lower:
        return True, "embedding/projection layers are sensitive"
    if "head" in lower:
        return True, "output head preserved for detail fidelity"
    if "lq_proj_in" in lower:
        return True, "LQ projection is special conditioning path"
    return False, "default robust layer"


def load_sensitivity_scores(path: str | Path | None) -> dict[str, float]:
    """Load optional sensitivity scores from a small JSON report.

    Accepted shapes:
    - {"layers": {"name": {"sensitivity": 0.9}}}
    - {"layers": {"name": {"score": 0.9}}}
    - {"layers": {"name": 0.9}}
    - {"name": 0.9}
    - [{"name": "layer", "sensitivity": 0.9}]
    """

    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "layers" in raw:
        raw = raw["layers"]

    scores: dict[str, float] = {}
    if isinstance(raw, dict):
        iterable = raw.items()
        for name, entry in iterable:
            if isinstance(entry, dict):
                value = entry.get("sensitivity", entry.get("score", entry.get("psnr_drop")))
            else:
                value = entry
            if value is not None:
                scores[str(name)] = float(value)
        return scores

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("Sensitivity list entries must contain a 'name' field")
            value = item.get("sensitivity", item.get("score", item.get("psnr_drop")))
            if value is None:
                raise ValueError(f"Sensitivity entry for {item['name']!r} is missing a score")
            scores[str(item["name"])] = float(value)
        return scores

    raise ValueError("Unsupported sensitivity JSON format")


def _validate_modes(default_mode: str, sensitive_mode: str, activation_qdq_mode: str) -> None:
    if default_mode not in VALID_LAYER_MODES - {"fp16_skip"}:
        raise ValueError(f"Unsupported default_mode: {default_mode}")
    if sensitive_mode not in VALID_LAYER_MODES:
        raise ValueError(f"Unsupported sensitive_mode: {sensitive_mode}")
    if activation_qdq_mode not in VALID_ACTIVATION_QDQ_MODES:
        raise ValueError(f"Unsupported activation_qdq_mode: {activation_qdq_mode}")


def _topk_sensitive_names(
    layer_names: Iterable[str],
    sensitivity_scores: dict[str, float],
    topk_sensitive: int,
) -> set[str]:
    if topk_sensitive <= 0 or not sensitivity_scores:
        return set()
    available = [(name, sensitivity_scores[name]) for name in layer_names if name in sensitivity_scores]
    available.sort(key=lambda item: item[1], reverse=True)
    return {name for name, _score in available[:topk_sensitive]}


def build_policy_from_layer_names(
    layer_names: Iterable[str],
    *,
    default_mode: str = "a8w8",
    sensitive_mode: str = "a16w8",
    activation_qdq_mode: str = "draq_symmetric",
    topk_sensitive: int = 0,
    sensitivity_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a PR4 mixed-precision policy from layer names plus optional scores."""

    _validate_modes(default_mode, sensitive_mode, activation_qdq_mode)
    names = list(layer_names)
    scores = sensitivity_scores or {}
    topk_names = _topk_sensitive_names(names, scores, topk_sensitive)

    layers: dict[str, dict[str, Any]] = {}
    counts = {default_mode: 0}
    if sensitive_mode != default_mode:
        counts[sensitive_mode] = 0

    for name in names:
        heuristic_sensitive, heuristic_reason = classify_layer_name(name)
        topk_sensitive_hit = name in topk_names
        if heuristic_sensitive or topk_sensitive_hit:
            if topk_sensitive_hit and heuristic_sensitive:
                reason = f"{heuristic_reason}; top-{topk_sensitive} sensitivity score"
            elif topk_sensitive_hit:
                reason = f"top-{topk_sensitive} sensitivity score"
            else:
                reason = heuristic_reason
            entry: dict[str, Any] = {"mode": sensitive_mode, "reason": reason}
            if name in scores:
                entry["sensitivity_score"] = scores[name]
            layers[name] = entry
            counts[sensitive_mode] = counts.get(sensitive_mode, 0) + 1
        else:
            counts[default_mode] = counts.get(default_mode, 0) + 1

    return {
        "schema_version": "flashvsr.pr4.layer_policy.v1",
        "name": "pr4_sensitivity_guided_mixed_precision",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "default": {"mode": default_mode, "activation_qdq_mode": activation_qdq_mode},
        "sensitive_mode": sensitive_mode,
        "sensitive_name_patterns": SENSITIVE_NAME_PATTERNS,
        "topk_sensitive": int(topk_sensitive),
        "counts": counts,
        "layers": layers,
    }


def list_linear_layer_names(model: Any) -> list[str]:
    """List all nn.Linear module names from a loaded DiT model."""

    import torch.nn as nn

    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def load_layer_names_from_json(path: str | Path) -> list[str]:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "layers" in raw:
        raw = raw["layers"]
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError("Layer names JSON must be a list of strings or {'layers': [...]} object")
    return list(raw)


def load_layer_names_from_checkpoint(checkpoint: Path) -> list[str]:
    """Load FlashVSR DiT and enumerate Linear layers from a checkpoint."""

    from scripts.ptq.fakequant_convert import build_dit, load_checkpoint

    model = build_dit()
    load_checkpoint(str(checkpoint), model)
    return list_linear_layer_names(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PR4 sensitivity-guided FlashVSR DiT PTQ layer policy")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="FP16 DiT checkpoint used to enumerate WanVideoDiT Linear layers")
    source.add_argument("--layer_names_json", type=Path, help="JSON list of Linear layer names; CPU-only fallback/test path")
    parser.add_argument("--output_policy", required=True, type=Path, help="Output policy JSON path")
    parser.add_argument("--default_mode", default="a8w8", choices=sorted(VALID_LAYER_MODES - {"fp16_skip"}))
    parser.add_argument("--sensitive_mode", default="a16w8", choices=sorted(VALID_LAYER_MODES))
    parser.add_argument("--activation_qdq_mode", default="draq_symmetric", choices=sorted(VALID_ACTIVATION_QDQ_MODES))
    parser.add_argument("--topk_sensitive", type=int, default=0, help="Mark top-k layers from --sensitivity_json as sensitive")
    parser.add_argument("--sensitivity_json", type=Path, default=None, help="Optional layer sensitivity score JSON")
    args = parser.parse_args()

    if args.layer_names_json:
        layer_names = load_layer_names_from_json(args.layer_names_json)
        source_path = str(args.layer_names_json)
    else:
        layer_names = load_layer_names_from_checkpoint(args.checkpoint)
        source_path = str(args.checkpoint)

    scores = load_sensitivity_scores(args.sensitivity_json)
    policy = build_policy_from_layer_names(
        layer_names,
        default_mode=args.default_mode,
        sensitive_mode=args.sensitive_mode,
        activation_qdq_mode=args.activation_qdq_mode,
        topk_sensitive=args.topk_sensitive,
        sensitivity_scores=scores,
    )
    policy["metadata"] = {
        "layer_name_source": source_path,
        "sensitivity_json": str(args.sensitivity_json) if args.sensitivity_json else None,
        "layer_count": len(layer_names),
        "sensitivity_score_count": len(scores),
    }

    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    args.output_policy.write_text(json.dumps(policy, indent=2))
    print(json.dumps({"output_policy": str(args.output_policy), "layers": len(layer_names), "counts": policy["counts"]}, indent=2))


if __name__ == "__main__":
    main()
