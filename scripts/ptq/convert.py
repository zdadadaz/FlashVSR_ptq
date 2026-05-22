"""
Convert FlashVSR WanModel to PTQ quantized format.

Supports:
- W8A16: Symmetric weight-only quantization (int8 weights, bf16 activations)
- W8A8: Symmetric weight + asymmetric activation quantization (int8 weights+activations)
"""

import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.wan_video_dit import WanModel
from src.models.quantization.ptq import (
    SymmetricWeightLinear,
    AsymmetricActLinear,
    convert_model_to_w8a16,
    convert_model_to_w8a8,
)


def load_model(checkpoint_path):
    """Load WanModel from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = WanModel(
        dim=1536,
        eps=1e-5,
        ffn_dim=6144,
        freq_dim=256,
        in_dim=16,
        num_heads=12,
        num_layers=30,
        out_dim=16,
        patch_size=(1, 2, 2),
        text_dim=4096,
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def load_calibration_cache(cache_path):
    """Load calibration cache from JSON."""
    with open(cache_path, "r") as f:
        cache_data = json.load(f)

    # Convert lists back to tensors
    act_stats = {}
    for name, stats in cache_data.items():
        act_stats[name] = (
            torch.tensor(stats["act_min"]),
            torch.tensor(stats["act_max"]),
        )
    return act_stats


def convert_model(model, act_stats, mode="w8a8", weight_method="max"):
    """Convert model to quantized version."""
    if mode == "w8a16":
        print("Converting to W8A16 (symmetric weight-only)...")
        model = convert_model_to_w8a16(model, method=weight_method)
    elif mode == "w8a8":
        print("Converting to W8A8 (symmetric weight + asymmetric activation)...")
        model = convert_model_to_w8a8(model, act_stats=act_stats, method=weight_method)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'w8a8' or 'w8a16'.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert FlashVSR model to PTQ quantized format")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to model .safetensors or .pth")
    parser.add_argument("--calibration_cache", type=str, required=True, help="Path to calibration cache JSON")
    parser.add_argument("--output_ptq", type=str, required=True, help="Path to save quantized model")
    parser.add_argument(
        "--mode",
        type=str,
        default="w8a8",
        choices=["w8a8", "w8a16"],
        help="Quantization mode: w8a8 (int8 weights + int8 activations) or w8a16 (int8 weights + bf16 activations)",
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        default="max",
        choices=["max", "percentile99", "std"],
        help="Weight quantization method",
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.input_ckpt)
    model.eval()

    # Load calibration cache
    act_stats = load_calibration_cache(args.calibration_cache)
    print(f"Loaded calibration stats for {len(act_stats)} layers")

    # Convert model
    model = convert_model(model, act_stats, mode=args.mode, weight_method=args.weight_method)

    # Save quantized model
    os.makedirs(os.path.dirname(args.output_ptq) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output_ptq)
    print(f"Quantized model saved to {args.output_ptq}")


if __name__ == "__main__":
    main()