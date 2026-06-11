"""
Convert FlashVSR DiT checkpoint → FakeQuant PTQ format (a8w8 / a16w8 / a8w4 / a16w4).

Steps:
  1. Load full-precision WanModel from checkpoint.
  2. Replace every nn.Linear with FakeQuantLinear (handles int4/int8 weight packing).
  3. Load calibration cache from calibrate step to set per-channel activation scales.
  4. Save the converted state_dict (or full model) to a .safetensors / .pth file.

The converted model runs via FakeQuantPipeline — no TensorRT required.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.wan_video_dit import WanModel
from src.models.quantization.fakequant import (
    ACTIVATION_QDQ_MODE_TO_ID,
    FakeQuantLinear,
    convert_model_to_fakequant,
)
from src.models.quantization.policy import layer_policy_entries, load_layer_policy


# =============================================================================
# Model builder
# =============================================================================

def build_dit(model_name: str = "FlashVSR-v1.1") -> WanModel:
    """Build WanModel with FlashVSR-v1.1 architecture."""
    return WanModel(
        dim=1536,
        eps=1e-5,
        ffn_dim=8960,
        freq_dim=256,
        in_dim=16,
        num_heads=12,
        num_layers=30,
        out_dim=16,
        patch_size=(1, 2, 2),
        text_dim=4096,
    )


def load_checkpoint(path: str, model: nn.Module):
    """Load state_dict, stripping 'model.' prefix."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[6:]] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return model


# =============================================================================
# Load calibration cache
# =============================================================================

def load_calibration_cache(cache_path: str, device="cuda"):
    """
    Load calibration cache from JSON.

    Returns dict: {layer_name: {'act_scale': tensor [Cin], 'zero_point': tensor [Cin]}}
    """
    if not os.path.exists(cache_path):
        return {}

    with open(cache_path, "r") as f:
        raw = json.load(f)

    result = {}
    for name, stats in raw.items():
        if name.startswith("_"):
            continue
        result[name] = {}
        if "act_scale" in stats:
            result[name]["act_scale"] = torch.tensor(stats["act_scale"], device=device)
        if "zero_point" in stats:
            result[name]["zero_point"] = torch.tensor(stats["zero_point"], device=device)
        if "act_mean" in stats:
            result[name]["act_mean"] = torch.tensor(stats["act_mean"], device=device)
        for key in (
            "draq_s_absmax",
            "draq_s_percentile_99",
            "draq_s_percentile_999",
            "draq_d_absmax",
            "draq_d_percentile_99",
            "draq_d_percentile_999",
        ):
            if key in stats:
                result[name][key] = torch.tensor(stats[key], device=device, dtype=torch.float32)
        if "draq_d_by_bucket" in stats:
            raw_buckets = stats["draq_d_by_bucket"]
            if isinstance(raw_buckets, dict):
                values = [raw_buckets[k] for k in sorted(raw_buckets, key=lambda item: str(item))]
            else:
                values = raw_buckets
            result[name]["draq_d_by_bucket"] = torch.tensor(values, device=device, dtype=torch.float32)
        if "mu_var" in stats:
            result[name]["mu_var"] = stats["mu_var"]
        if "volts_tier" in stats:
            result[name]["volts_tier"] = stats["volts_tier"]
    return result


def load_smoothquant_cache(cache_path: str, device="cuda"):
    """Load per-layer SmoothQuant migration scales from JSON.

    Accepted entry shapes:
      {"layer": {"smoothquant_scale": [...]}}
      {"layer": {"scale": [...]}}
      {"layer": [...]}
    """
    if not cache_path or not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        raw = json.load(f)
    result = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        value = entry
        if isinstance(entry, dict):
            value = entry.get("smoothquant_scale", entry.get("scale"))
        if value is None:
            continue
        result[name] = torch.tensor(value, device=device, dtype=torch.float32)
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert FlashVSR DiT → FakeQuant PTQ format")
    parser.add_argument("--checkpoint",       type=str, required=True,
                       help="Path to full-precision DiT .safetensors or .pth")
    parser.add_argument("--calibration_cache", type=str, default="",
                       help="Path to calibration JSON cache from fakequant_calibrate.py")
    parser.add_argument("--output",          type=str, required=True,
                       help="Output path for quantized checkpoint")
    parser.add_argument(
        "--mode", type=str, default="a8w8",
        choices=["a16w8", "a8w8", "a16w4", "a8w4"],
        help="Quantization mode"
    )
    parser.add_argument(
        "--static_quality_policy", type=str, default="none",
        choices=["none", "sensitive_a16", "self_attn_only_a8"],
        help=(
            "Static PTQ quality policy. 'sensitive_a16' keeps A8W8 checkpoint "
            "structure but disables activation QDQ for text/time/projection/head/FFN "
            "layers. 'self_attn_only_a8' keeps static A8 activation QDQ only on "
            "self-attention projections. Both preserve int8 weights."
        ),
    )
    parser.add_argument(
        "--activation_qdq_mode", type=str, default="static_asymmetric",
        choices=list(ACTIVATION_QDQ_MODE_TO_ID),
        help=(
            "A8 activation QDQ policy. static_asymmetric uses calibrated per-channel "
            "scale/zero_point from --calibration_cache. dynamic_symmetric, "
            "dynamic_asymmetric and draq_symmetric compute activation scales at runtime. "
            "draq_static_s, draq_static_sd_layer and draq_static_sd_bucket use "
            "calibration-derived DRAQ static fields."
        ),
    )
    parser.add_argument(
        "--policy_json", type=str, default="",
        help="Optional per-layer policy JSON for mixed precision recovery."
    )
    parser.add_argument(
        "--enable_bias_correction", action="store_true",
        help="Apply activation-mean-based deterministic bias correction when act_mean exists in calibration cache."
    )
    parser.add_argument(
        "--smoothquant_cache", type=str, default="",
        help="Optional JSON containing per-layer SmoothQuant migration scales."
    )
    parser.add_argument(
        "--weight_rounding", type=str, default="nearest", choices=["nearest", "adaround"],
        help="Weight rounding method. 'adaround' uses calibration act_mean for deterministic AdaRound-lite rounding."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load full-precision model
    # ------------------------------------------------------------------
    print(f"\n[Convert] Loading checkpoint: {args.checkpoint}")
    model = build_dit()
    model = load_checkpoint(args.checkpoint, model)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Load calibration cache (activation scales)
    # ------------------------------------------------------------------
    act_stats = {}
    if args.calibration_cache:
        act_stats = load_calibration_cache(args.calibration_cache)
        print(f"[Convert] Loaded calibration for {len(act_stats)} layers")
    static_cache_modes = {"static_asymmetric", "draq_static_s", "draq_static_sd_layer", "draq_static_sd_bucket"}
    if args.mode.startswith("a8") and args.activation_qdq_mode in static_cache_modes and not act_stats:
        raise RuntimeError(
            f"Mode {args.mode} with {args.activation_qdq_mode} activation QDQ requires a non-empty "
            "--calibration_cache with calibrated activation entries."
        )

    layer_policy = None
    if args.policy_json:
        layer_policy = layer_policy_entries(load_layer_policy(args.policy_json))
        print(f"[Convert] Loaded layer policy for {len(layer_policy)} layers: {args.policy_json}")

    smoothquant_scales = {}
    if args.smoothquant_cache:
        smoothquant_scales = load_smoothquant_cache(args.smoothquant_cache)
        print(f"[Convert] Loaded SmoothQuant scales for {len(smoothquant_scales)} layers")

    # ------------------------------------------------------------------
    # 3. Convert nn.Linear → FakeQuantLinear
    # ------------------------------------------------------------------
    print(f"\n[Convert] Converting to {args.mode} …")
    model = convert_model_to_fakequant(
        model,
        mode=args.mode,
        act_stats=act_stats,
        static_quality_policy=args.static_quality_policy,
        activation_qdq_mode=args.activation_qdq_mode,
        layer_policy=layer_policy,
        enable_bias_correction=args.enable_bias_correction,
        smoothquant_scales=smoothquant_scales,
        weight_rounding=args.weight_rounding,
    )

    # ------------------------------------------------------------------
    # 4. Save converted model
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    sd = model.state_dict()
    if args.output.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(sd, args.output)
    else:
        torch.save(sd, args.output)

    total_int  = sum(1 for v in sd.values() if v.dtype in (torch.int8, torch.int32))
    total_params = sum(1 for v in sd.values() if torch.is_floating_point(v))
    print(f"\n[Convert] Saved → {args.output}")
    print(f"[Convert] Total tensors: {len(sd)}  float={total_params}  int={total_int}")
    disabled = sum(
        1 for k, v in sd.items()
        if k.endswith("act_quant_enabled") and hasattr(v, "item") and not bool(v.item())
    )
    enabled = sum(
        1 for k, v in sd.items()
        if k.endswith("act_quant_enabled") and hasattr(v, "item") and bool(v.item())
    )
    print(f"[Convert] act_quant_enabled: enabled={enabled} disabled={disabled}")

    summary = dict(getattr(model, "_fakequant_conversion_summary", {}))
    summary.update({
        "checkpoint": args.checkpoint,
        "output": args.output,
        "calibration_cache": args.calibration_cache or None,
        "policy_json": args.policy_json or None,
        "smoothquant_cache": args.smoothquant_cache or None,
        "weight_rounding": args.weight_rounding,
    })
    summary_path = f"{args.output}.conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Convert] Summary → {summary_path}")


if __name__ == "__main__":
    main()
