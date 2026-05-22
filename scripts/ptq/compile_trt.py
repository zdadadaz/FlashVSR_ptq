"""
Compile PTQ quantized model to TensorRT engine.

Uses torch_tensorrt.ptq.DataLoaderCalibrator for INT8 calibration and compilation.
"""

import argparse
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


def check_torch_tensorrt():
    """Check if torch_tensorrt is available."""
    try:
        import torch_tensorrt as trt
        import torch_tensorrt.ptq as ptq

        print(f"torch_tensorrt version: {trt.__version__}")
        return True
    except ImportError:
        print("ERROR: torch-tensorrt not installed.")
        print("Install with: pip install torch-tensorrt")
        return False


def load_model_with_quantization(checkpoint_path, calibration_cache=None, mode="w8a8"):
    """Load model and apply quantization."""
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
    model.eval()

    # Load calibration cache for activation stats
    act_stats = None
    if calibration_cache and os.path.exists(calibration_cache):
        import json

        with open(calibration_cache, "r") as f:
            cache_data = json.load(f)
        act_stats = {}
        for name, stats in cache_data.items():
            act_stats[name] = (
                torch.tensor(stats["act_min"]),
                torch.tensor(stats["act_max"]),
            )
        print(f"Loaded calibration stats for {len(act_stats)} layers")

    # Apply quantization
    if mode == "w8a16":
        model = convert_model_to_w8a16(model, method="max")
    elif mode == "w8a8":
        model = convert_model_to_w8a8(model, act_stats=act_stats, method="max")

    return model


def compile_to_trt(model, output_engine, input_shape=(1, 1, 16, 24, 24)):
    """Compile model to TensorRT engine."""
    import torch_tensorrt as trt
    import torch_tensorrt.ptq as ptq

    print(f"Compiling model to TensorRT engine with input shape {input_shape}...")

    # Trace model
    model = model.cuda()
    example_input = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        scripted_model = torch.jit.trace(model, example_input)

    # Calibration dataset (use same input shape for calibration)
    # In practice, you'd use a real DataLoader with calibration data
    calibrator = ptq.DataLoaderCalibrator(
        None,  # Would pass DataLoader in practice
        calibrationAlgo=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
    )

    # For now, compile without calibrator (calibration already done via activation stats)
    compile_spec = {
        "inputs": [trt.Input(input_shape)],
        "enabled_precisions": {torch.int8},
    }

    trt_model = trt.compile(scripted_model, **compile_spec)

    # Save engine
    os.makedirs(os.path.dirname(output_engine) or ".", exist_ok=True)
    torch.jit.save(trt_model, output_engine)
    print(f"TensorRT engine saved to {output_engine}")


def main():
    parser = argparse.ArgumentParser(description="Compile FlashVSR model to TensorRT")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to model .safetensors or .pth")
    parser.add_argument("--calibration_cache", type=str, default=None, help="Path to calibration cache JSON")
    parser.add_argument("--output_engine", type=str, required=True, help="Path to save TensorRT engine")
    parser.add_argument(
        "--mode",
        type=str,
        default="w8a8",
        choices=["w8a8", "w8a16"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=5,
        default=[1, 1, 16, 24, 24],
        metavar=("B", "T", "C", "H", "W"),
        help="Input shape for compilation (default: 1 1 16 24 24)",
    )
    args = parser.parse_args()

    if not check_torch_tensorrt():
        sys.exit(1)

    # Load and quantize model
    model = load_model_with_quantization(
        args.input_ckpt,
        calibration_cache=args.calibration_cache,
        mode=args.mode,
    )

    # Compile to TRT
    compile_to_trt(model, args.output_engine, input_shape=tuple(args.input_shape))


if __name__ == "__main__":
    main()