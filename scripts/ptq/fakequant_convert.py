"""
Convert FlashVSR DiT checkpoint → FakeQuant PTQ format (a8w8 / a16w8 / a8w4 / a16w4 / a4w4).

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
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.wan_video_dit import WanModel
from src.models.quantization.fakequant import (
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
        result[name] = {
            "act_scale": torch.tensor(stats["act_scale"], device=device),
            "zero_point": torch.tensor(stats["zero_point"], device=device),
        }
        if "act_mean" in stats:
            result[name]["act_mean"] = torch.tensor(stats["act_mean"], device=device)
    return result


def load_lsgquant_layer_policy(path: str | Path):
    """Load PR-3 LSGQuant policy entries plus a compact summary for manifests."""

    raw = load_layer_policy(path)
    entries = layer_policy_entries(raw)
    tier_counts = {}
    mode_counts = {}
    for entry in entries.values():
        tier = entry.get("tier")
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        mode = entry.get("mode")
        if mode:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
    summary = {
        "path": str(path),
        "schema_version": raw.get("schema_version"),
        "scope": raw.get("scope"),
        "default": raw.get("default"),
        "thresholds": raw.get("thresholds"),
        "tier_counts": tier_counts,
        "mode_counts": mode_counts,
        "layers": len(entries),
    }
    return entries, summary


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
        choices=["a16w8", "a8w8", "a16w4", "a8w4", "a4w4"],
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
        choices=["static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric", "draq_symmetric"],
        help=(
            "Activation QDQ policy. A8 static_asymmetric uses calibrated per-channel "
            "scale/zero_point from --calibration_cache. dynamic_symmetric and "
            "dynamic_asymmetric compute per-token activation scales at runtime; "
            "draq_symmetric uses LSGQuant online channel+token scaling."
        ),
    )
    parser.add_argument(
        "--draq_qrange", type=str, default="signed_symmetric",
        choices=["signed_symmetric", "signed_full"],
        help="DRAQ signed int8 clamp range: conservative [-127,127] or paper-style [-128,127]."
    )
    parser.add_argument(
        "--policy_json", type=str, default="",
        help="Optional per-layer policy JSON for mixed precision recovery."
    )
    parser.add_argument(
        "--policy", type=str, default="",
        help="Alias for --policy_json; intended for LSGQuant/VOLTS PR-3 policy files."
    )
    parser.add_argument(
        "--enable_bias_correction", action="store_true",
        help="Apply activation-mean-based deterministic bias correction when act_mean exists in calibration cache."
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
    if args.mode.startswith("a8") and args.activation_qdq_mode == "static_asymmetric" and not act_stats:
        raise RuntimeError(
            f"Mode {args.mode} with static_asymmetric activation QDQ requires a non-empty "
            "--calibration_cache with calibrated act_scale and zero_point entries."
        )

    layer_policy = None
    policy_summary = None
    policy_path = args.policy or args.policy_json
    if args.policy and args.policy_json and args.policy != args.policy_json:
        raise ValueError("Use only one policy path: --policy or --policy_json")
    if policy_path:
        layer_policy, policy_summary = load_lsgquant_layer_policy(policy_path)
        print(f"[Convert] Loaded layer policy for {len(layer_policy)} layers: {policy_path}")

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
        draq_qrange=args.draq_qrange,
        layer_policy=layer_policy,
        enable_bias_correction=args.enable_bias_correction,
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
        "policy_json": policy_path or None,
        "policy_summary": policy_summary,
        "draq_qrange": args.draq_qrange,
    })
    summary_path = f"{args.output}.conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Convert] Summary → {summary_path}")


if __name__ == "__main__":
    main()
