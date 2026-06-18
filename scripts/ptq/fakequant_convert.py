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
    Also returns `act_min` / `act_max` so callers can recompute alternative
    quantization schemes (e.g. symmetric scales for mode 7).
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
        # Also persist act_min / act_max so we can recompute symmetric scales
        # for activation_qdq_mode=static_tensor_symmetric (mode 7). The stored
        # act_scale is the asymmetric per-channel scale; for symmetric int8 we
        # need scale = max(|act_min|, |act_max|) / 127 with zero_point=0.
        if "act_min" in stats:
            result[name]["act_min"] = torch.tensor(stats["act_min"], device=device, dtype=torch.float32)
        if "act_max" in stats:
            result[name]["act_max"] = torch.tensor(stats["act_max"], device=device, dtype=torch.float32)
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
        # Output QDQ fields (per-output-channel symmetric) used by mode 7
        # (static_tensor_symmetric). Loaded into a separate dict by
        # load_output_calibration_cache.
        if "output_scale" in stats:
            result[name]["output_scale"] = torch.tensor(stats["output_scale"], device=device, dtype=torch.float32)
        if "output_zero_point" in stats:
            result[name]["output_zero_point"] = torch.tensor(stats["output_zero_point"], device=device, dtype=torch.int32)
    return result


def load_output_calibration_cache(act_stats: dict, device: str = "cuda") -> dict:
    """Pull per-layer output_scale / output_zero_point from a loaded calibration cache.

    Returns a dict compatible with FakeQuantLinear.from_float(output_scale=, output_zero_point=)
    contract: {layer_name: {"output_scale": Tensor[out_features], "output_zero_point": Tensor[out_features]}}.
    """
    out = {}
    for name, stats in act_stats.items():
        if "output_scale" in stats and "output_zero_point" in stats:
            out[name] = {
                "output_scale": stats["output_scale"],
                "output_zero_point": stats["output_zero_point"],
            }
    return out


def recompute_symmetric_act_scales(act_stats: dict) -> dict:
    """Recompute per-channel symmetric int8 act_scale from act_min/act_max.

    The calibration cache stores asymmetric per-channel scale + zero_point
    (`act_scale = (max-min)/255`, `zero_point = -min/scale`). For
    ``activation_qdq_mode=static_tensor_symmetric`` (mode 7) we need a
    symmetric per-channel scale with zero_point=0:

        sym_scale = max(|act_min|, |act_max|) / 127

    Returns a NEW dict in the same shape as ``act_stats`` with `act_scale`
    and `zero_point` overridden, leaving `act_min`/`act_max` etc. intact for
    debugging.
    """
    out = {}
    for name, s in act_stats.items():
        if "act_min" not in s or "act_max" not in s:
            out[name] = s
            continue
        act_min = s["act_min"]
        act_max = s["act_max"]
        # Move to cpu to compute (avoid device-specific shape quirks)
        sym_scale = torch.maximum(act_min.abs(), act_max.abs()) / 127.0
        sym_scale = sym_scale.clamp(min=1e-6)
        s_new = dict(s)
        s_new["act_scale"] = sym_scale.to(s["act_scale"].device, s["act_scale"].dtype)
        s_new["zero_point"] = torch.zeros_like(s["zero_point"])
        out[name] = s_new
    return out


def rescale_output_stats(output_stats: dict, act_stats: dict, multiplier: float = 1.5) -> dict:
    """Widen per-output-channel output_scale by a safety multiplier.

    FlashVSR DiT's per-output-channel distributions have heavy tails that aren't
    fully captured by short calibration runs (8-32 samples). Without a multiplier
    the static bound clips up to 30%+ of channels at inference. 1.5x is a
    conservative default that preserves headroom for inter-timestep dynamic
    range while keeping int8 effective bits reasonable.

    Only the `output_scale` is scaled (NOT the act_scale) — act_scale is the
    input-side bound which is naturally tighter; output_scale is the bound
    applied to F.linear results, where per-channel outliers dominate.
    """
    if multiplier == 1.0:
        return output_stats
    out = {}
    for name, entry in output_stats.items():
        scale = entry["output_scale"]
        # Ensure fp32
        scale = (scale.float() * float(multiplier)).clamp(min=1e-8)
        out[name] = {
            "output_scale": scale.to(entry["output_scale"].device, entry["output_scale"].dtype),
            "output_zero_point": entry["output_zero_point"],
        }
    return out


def collapse_output_stats_to_per_tensor(output_stats: dict) -> dict:
    """Collapse per-output-channel output_scale to a single per-layer scalar.

    The PTQ project's FlashVSR_PTQ uses per-tensor (scalar) output_scale for
    robustness against per-channel outliers. This is the contract that
    produces stable 29.99 dB in their a8w8 static run.

    The collapse rule: per-tensor scale = max(all per-channel scales) / 127
    with broadcast to scalar — i.e. one value per layer. The forward path
    treats this as a per-tensor bound (scale.numel() == 1 → no per-channel
    division in _qdq_symmetric_channel).
    """
    out = {}
    for name, entry in output_stats.items():
        scale = entry["output_scale"].float()
        # Take the max across output channels as the per-tensor bound.
        scalar = scale.amax().clamp(min=1e-8)
        out[name] = {
            "output_scale": scalar.reshape(1),  # 1-element tensor → per-tensor
            "output_zero_point": entry["output_zero_point"][:1].clone() if entry["output_zero_point"].numel() > 1 else entry["output_zero_point"],
        }
    return out


def collapse_act_stats_to_per_tensor(act_stats: dict) -> dict:
    """Collapse per-channel act_scale to per-tensor (scalar) for symmetric mode 7.

    Same rationale as collapse_output_stats_to_per_tensor. The cache stores
    per-channel scales (asymmetric min/max ranges); for symmetric int8 we
    use the GLOBAL max(|act_min|, |act_max|) / 127 as a single scalar per
    layer. This is much more robust to outlier channels that aren't fully
    captured by short calibration.

    Output shape: act_scale with 1 element. The runtime QDQ path in
    FakeQuantLinear.forward detects numel()==1 and treats it as per-tensor
    (no per-channel broadcast).
    """
    out = {}
    for name, s in act_stats.items():
        if "act_min" not in s or "act_max" not in s:
            out[name] = s
            continue
        act_min = s["act_min"].float()
        act_max = s["act_max"].float()
        sym_scale = torch.maximum(act_min.abs(), act_max.abs()).amax() / 127.0
        sym_scale = sym_scale.clamp(min=1e-6)
        s_new = dict(s)
        # IMPORTANT: shape must be (1, 1, 1) so numel()==1 in the forward path
        # and broadcasts to whatever in_features the layer has.
        if s["act_scale"].dim() == 3:
            target_shape = (1, 1, 1)
        elif s["act_scale"].dim() == 1:
            target_shape = (1,)
        else:
            target_shape = (1,) * s["act_scale"].dim()
        s_new["act_scale"] = sym_scale.to(s["act_scale"].device, s["act_scale"].dtype).reshape(target_shape)
        s_new["zero_point"] = torch.zeros(target_shape, dtype=s["zero_point"].dtype, device=s["zero_point"].device)
        out[name] = s_new
    return out


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
        choices=list(ACTIVATION_QDQ_MODE_TO_ID),
        help=(
            "Activation QDQ policy. A8 static_asymmetric uses calibrated per-channel "
            "scale/zero_point from --calibration_cache. dynamic_symmetric and "
            "dynamic_asymmetric compute per-token activation scales at runtime; "
            "draq_symmetric uses LSGQuant online channel+token scaling. "
            "draq_static_s, draq_static_sd_layer and draq_static_sd_bucket use "
            "calibration-derived DRAQ static fields."
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
    parser.add_argument(
        "--smoothquant_cache", type=str, default="",
        help="Optional JSON containing per-layer SmoothQuant migration scales."
    )
    parser.add_argument(
        "--enable_smoothquant", action="store_true",
        help="Opt-in: apply SmoothQuant scale even when --smoothquant_cache is provided. "
             "Default off — DMQ (ICCV'25) evidence shows hand-crafted SmoothQuant hurts "
             "diffusion PSNR without learned scaling."
    )
    parser.add_argument(
        "--output_scale_multiplier", type=float, default=1.5,
        help="Per-output-channel safety multiplier for static output QDQ (mode 7). "
             "FlashVSR DiT has heavy per-channel outliers not fully captured by short "
             "calibration. 1.5x is the production-safe default; 1.0 disables."
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
    policy_summary = None
    policy_path = args.policy or args.policy_json
    if args.policy and args.policy_json and args.policy != args.policy_json:
        raise ValueError("Use only one policy path: --policy or --policy_json")
    if policy_path:
        layer_policy, policy_summary = load_lsgquant_layer_policy(policy_path)
        print(f"[Convert] Loaded layer policy for {len(layer_policy)} layers: {policy_path}")

    smoothquant_scales = {}
    if args.smoothquant_cache:
        smoothquant_scales = load_smoothquant_cache(args.smoothquant_cache)
        print(f"[Convert] Loaded SmoothQuant scales for {len(smoothquant_scales)} layers")

    output_stats = {}
    if act_stats and args.activation_qdq_mode == "static_tensor_symmetric":
        # Recompute act_scale as per-channel symmetric: max(|min|, |max|)/127
        # The cache stores asymmetric per-channel scale + zero_point; symmetric
        # mode 7 requires zero_point=0, so reusing the asymmetric scale here
        # causes severe clamping (verified: produced 12.7 dB on bowing_cif).
        act_stats = recompute_symmetric_act_scales(act_stats)
        output_stats = load_output_calibration_cache(act_stats)
        # Apply safety multiplier to widen the per-output-channel bound.
        # Calibration with 8-32 samples doesn't capture full inter-timestep
        # dynamic range; without this 30%+ of channels clip at inference.
        if args.output_scale_multiplier != 1.0:
            output_stats = rescale_output_stats(output_stats, act_stats, args.output_scale_multiplier)
            print(f"[Convert] Loaded output QDQ scales for {len(output_stats)} layers (mode=7, symmetric act scale, output_scale × {args.output_scale_multiplier})")
        else:
            print(f"[Convert] Loaded output QDQ scales for {len(output_stats)} layers (mode=7, symmetric act scale)")

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
        smoothquant_scales=smoothquant_scales,
        enable_smoothquant=args.enable_smoothquant,
        weight_rounding=args.weight_rounding,
        output_stats=output_stats,
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
        "smoothquant_cache": args.smoothquant_cache or None,
        "weight_rounding": args.weight_rounding,
    })
    summary_path = f"{args.output}.conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Convert] Summary → {summary_path}")


if __name__ == "__main__":
    main()
