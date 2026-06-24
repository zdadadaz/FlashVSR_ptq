"""Static PTQ FakeQuant baseline runner for FlashVSR DiT.

Builds the requested reproducible baseline:
- static asymmetric A8W8 FakeQuant for DiT Linear layers;
- SmoothQuant channel migration scales from calibration stats + FP weights;
- AdaRound-lite weight rounding through fakequant_convert.py;
- 10-20% sensitive Linear layers fallback to FP16 (`fp16_skip` policy);
- optional FP16-vs-PTQ PSNR command execution.

Wan VAE/decoder/pipeline non-DiT modules remain out of quantization scope.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ptq.fakequant_convert import build_dit, load_checkpoint  # noqa: E402
from src.models.quantization.policy import classify_layer_name  # noqa: E402


def _flatten_floats(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_floats(item))
        return out
    return [float(value)]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def layer_names_from_model(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def sensitivity_score(name: str, stats: dict[str, Any] | None) -> float:
    """Heuristic static PTQ sensitivity score from activation dynamic range.

    This is intentionally deterministic and calibration-cache only.  Groups known
    to be brittle under static A8 receive a small multiplier so the top 10-20%
    fallback set is stable even when ranges are close.
    """
    group = classify_layer_name(name)
    group_weight = {
        "time": 1.35,
        "embed": 1.25,
        "head": 1.20,
        "ffn": 1.15,
        "other": 1.10,
        "cross_attn": 1.00,
        "self_attn": 0.95,
    }.get(group, 1.0)
    if not stats:
        return group_weight
    mins = _flatten_floats(stats.get("act_min"))
    maxs = _flatten_floats(stats.get("act_max"))
    if mins and maxs:
        dyn_range = max(maxs) - min(mins)
        amax = max(abs(min(mins)), abs(max(maxs)))
        return group_weight * (float(dyn_range) + 0.01 * float(amax))
    scales = _flatten_floats(stats.get("act_scale", stats.get("scale")))
    if scales:
        return group_weight * max(scales)
    return group_weight


def build_static_mixed_policy(
    layer_names: list[str],
    calibration: dict[str, Any],
    fallback_ratio: float = 0.15,
) -> dict[str, Any]:
    if not 0.0 <= fallback_ratio <= 1.0:
        raise ValueError("fallback_ratio must be in [0, 1]")
    fallback_count = int(math.ceil(len(layer_names) * fallback_ratio))
    ranked = sorted(
        layer_names,
        key=lambda name: (sensitivity_score(name, calibration.get(name)), name),
        reverse=True,
    )
    fallback = set(ranked[:fallback_count])
    layers: dict[str, dict[str, str]] = {}
    counts = {"fp16_skip": 0, "a8w8": 0}
    for name in layer_names:
        group = classify_layer_name(name)
        if name in fallback:
            layers[name] = {
                "mode": "fp16_skip",
                "group": group,
                "reason": f"top {fallback_ratio:.0%} static-A8 sensitivity fallback to FP16",
            }
            counts["fp16_skip"] += 1
        else:
            layers[name] = {
                "mode": "a8w8",
                "activation_qdq_mode": "static_asymmetric",
                "group": group,
                "reason": "static A8W8 baseline layer",
            }
            counts["a8w8"] += 1
    return {
        "schema_version": "flashvsr.fakequant.layer_policy.v1",
        "name": f"static_a8w8_sq_adaround_fp16fallback_{int(fallback_ratio * 100)}pct",
        "scope": "WanVideoDiT Linear layers only; Wan VAE remains unquantized",
        "default": {"mode": "a8w8", "activation_qdq_mode": "static_asymmetric"},
        "fallback_ratio": fallback_ratio,
        "counts": counts,
        "layers": layers,
    }


def _act_amax(stats: dict[str, Any]) -> torch.Tensor | None:
    mins = _flatten_floats(stats.get("act_min"))
    maxs = _flatten_floats(stats.get("act_max"))
    if mins and maxs and len(mins) == len(maxs):
        return torch.maximum(torch.tensor(mins).abs(), torch.tensor(maxs).abs()).float()
    scales = _flatten_floats(stats.get("act_scale", stats.get("scale")))
    if scales:
        # qrange ~= 255 for signed asymmetric static A8.  This is a fallback for
        # older caches that did not persist min/max.
        return torch.tensor(scales).float().abs() * 255.0
    return None


def build_smoothquant_cache(
    model: nn.Module,
    calibration: dict[str, Any],
    alpha: float = 0.5,
    clamp_min: float = 1e-5,
    clamp_max: float = 1e5,
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    out: dict[str, Any] = {
        "_metadata": {
            "schema_version": "flashvsr.smoothquant_cache.v1",
            "alpha": alpha,
            "formula": "scale = act_amax^alpha / weight_amax_input^(1-alpha)",
        }
    }
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in calibration:
            continue
        act_amax = _act_amax(calibration[name])
        if act_amax is None:
            continue
        if act_amax.numel() == 1:
            act_amax = act_amax.expand(module.in_features)
        if act_amax.numel() != module.in_features:
            continue
        weight = module.weight.detach().to(torch.float32)
        weight_amax = torch.amax(torch.abs(weight), dim=0).clamp(min=1e-8)
        scale = (act_amax.clamp(min=1e-8).pow(alpha) / weight_amax.pow(1.0 - alpha)).clamp(
            min=clamp_min, max=clamp_max
        )
        out[name] = {"smoothquant_scale": [float(x) for x in scale.cpu().tolist()]}
    return out



def load_les_cache(path: str | Path) -> dict[str, Any]:
    """Load Learned Equivalent Scaling tau cache as SmoothQuant-compatible scales.

    Accepted entry shapes:
      {"layer": {"tau": [...]}}
      {"layer": {"smoothquant_scale": [...]}}
      {"layer": [...]}
    Returns a JSON-serializable SmoothQuant cache so fakequant_convert.py can reuse
    the existing --smoothquant_cache path without runtime changes.
    """
    raw = load_json(path)
    out: dict[str, Any] = {
        "_metadata": {
            "schema_version": "flashvsr.smoothquant_cache.v1",
            "source_schema": raw.get("_metadata", {}).get("schema_version", "flashvsr.learned_equivalent_scaling.v1"),
            "source": str(path),
            "scale_semantics": "LES tau loaded through SmoothQuant-compatible buffer",
        }
    }
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        value = entry
        if isinstance(entry, dict):
            value = entry.get("tau", entry.get("smoothquant_scale", entry.get("scale")))
        if value is None:
            continue
        out[name] = {"smoothquant_scale": [float(x) for x in _flatten_floats(value)]}
    return out


def select_ptq_policy_and_scales(
    model: nn.Module,
    layer_names: list[str],
    calibration: dict[str, Any],
    fallback_ratio: float,
    smoothquant_alpha: float,
    sensitivity_cache: str | None = None,
    les_cache: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select policy/scales for static baseline, true-sensitivity E, or LES B.

    E: --sensitivity_cache overrides the old heuristic with measured MSE ranking
    and uses dynamic_asymmetric for non-skipped layers as requested by the daily
    master schedule. B: --les_cache supplies learned tau through the existing
    SmoothQuant scale buffer.
    """
    if sensitivity_cache:
        from scripts.ptq.build_true_sensitive_policy import build_true_sensitive_policy, load_sensitivity_cache

        policy = build_true_sensitive_policy(
            load_sensitivity_cache(sensitivity_cache),
            fp16_skip_ratio=fallback_ratio,
            robust_mode="a8w8",
            robust_activation_qdq_mode="dynamic_asymmetric",
        )
    else:
        policy = build_static_mixed_policy(layer_names, calibration, fallback_ratio)

    smoothquant_cache = load_les_cache(les_cache) if les_cache else build_smoothquant_cache(model, calibration, alpha=smoothquant_alpha)
    return policy, smoothquant_cache

def run_cmd(cmd: Iterable[str], dry_run: bool) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"$ {printable}")
    if not dry_run:
        subprocess.run(list(cmd), cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlashVSR static A8W8 + SmoothQuant + AdaRound PTQ baseline")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration_cache", required=True)
    parser.add_argument("--out_dir", default="outputs/static_ptq_baseline")
    parser.add_argument("--fallback_ratio", type=float, default=0.15, help="Sensitive FP16 fallback ratio; use 0.10-0.20")
    parser.add_argument("--smoothquant_alpha", type=float, default=0.5)
    parser.add_argument("--sensitivity_cache", type=str, default=None, help="Build policy from true per-layer sensitivity cache (Direction E)")
    parser.add_argument("--les_cache", type=str, default=None, help="Use Learned Equivalent Scaling tau cache as SmoothQuant-compatible scales (Direction B)")
    parser.add_argument("--fp16_video", default="", help="Optional already-rendered FP16 output video")
    parser.add_argument("--ptq_video", default="", help="Optional already-rendered PTQ output video")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not 0.05 <= args.fallback_ratio <= 0.40:
        raise ValueError("Use --fallback_ratio between 0.05 and 0.40 for heuristic/true-sensitivity fallback")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_json(args.calibration_cache)

    model = build_dit()
    if not args.dry_run:
        model = load_checkpoint(args.checkpoint, model)
    layer_names = layer_names_from_model(model)

    policy, smoothquant_cache = select_ptq_policy_and_scales(
        model=model,
        layer_names=layer_names,
        calibration=calibration,
        fallback_ratio=args.fallback_ratio,
        smoothquant_alpha=args.smoothquant_alpha,
        sensitivity_cache=args.sensitivity_cache,
        les_cache=args.les_cache,
    )
    policy_path = out_dir / "static_mixed_fp16fallback_policy.json"
    policy_path.write_text(json.dumps(policy, indent=2))

    smoothquant_path = out_dir / "smoothquant_scales.json"
    smoothquant_path.write_text(json.dumps(smoothquant_cache, indent=2))

    ckpt_out = out_dir / "dit_static_a8w8_sq_adaround_mixed.safetensors"
    convert_cmd = [
        sys.executable,
        "scripts/ptq/fakequant_convert.py",
        "--checkpoint", args.checkpoint,
        "--calibration_cache", args.calibration_cache,
        "--output", str(ckpt_out),
        "--mode", "a8w8",
        "--activation_qdq_mode", "static_asymmetric",
        "--policy_json", str(policy_path),
        "--smoothquant_cache", str(smoothquant_path),
        "--weight_rounding", "adaround",
        "--enable_bias_correction",
    ]
    run_cmd(convert_cmd, args.dry_run)

    psnr_json = None
    if args.fp16_video and args.ptq_video:
        psnr_json = out_dir / "psnr_fp16_vs_static_ptq.json"
        psnr_cmd = [sys.executable, "scripts/compare_video_psnr.py", args.fp16_video, args.ptq_video, "--out-json", str(psnr_json)]
        run_cmd(psnr_cmd, args.dry_run)

    manifest = {
        "schema_version": "flashvsr.static_ptq_baseline.v1",
        "checkpoint": args.checkpoint,
        "calibration_cache": args.calibration_cache,
        "out_dir": str(out_dir),
        "policy_json": str(policy_path),
        "smoothquant_cache": str(smoothquant_path),
        "output_checkpoint": str(ckpt_out),
        "fallback_ratio": args.fallback_ratio,
        "smoothquant_alpha": args.smoothquant_alpha,
        "sensitivity_cache": args.sensitivity_cache,
        "les_cache": args.les_cache,
        "weight_rounding": "adaround",
        "convert_cmd": convert_cmd,
        "psnr_json": str(psnr_json) if psnr_json else None,
        "counts": policy["counts"],
    }
    manifest_path = out_dir / "static_ptq_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[StaticPTQBaseline] manifest → {manifest_path}")


if __name__ == "__main__":
    main()
