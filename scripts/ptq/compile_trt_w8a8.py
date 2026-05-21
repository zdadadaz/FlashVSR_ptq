"""TensorRT INT8 compilation script for FlashVSR DiT.

Compiles DiT model with RMSNorm folding via:
1. torch.export to capture clean graph
2. torch_tensorrt.ptq.DataLoaderCalibrator with INT8 calibration
3. torch_tensorrt.compile for TensorRT engine generation

Supports dynamic shape ranges: T:1-64, H/W:128-2048, B:1-8.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.export

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.wan_video_dit import WanModel
from src.models.quantization.rmsnorm_fold import fold_dit_rmsnorms

# Optional: torch_tensorrt
try:
    import torch_tensorrt as trt
    import torch_tensorrt.ptq as ptq
    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False
    trt = None
    ptq = None


def check_torch_tensorrt():
    """Check if torch-tensorrt is available.

    Returns:
        bool: True if torch_tensorrt is importable, False otherwise.
    """
    if not TORCH_TENSORRT_AVAILABLE:
        print("ERROR: torch-tensorrt is not installed.")
        print("Install with: pip install torch-tensorrt")
        return False
    print(f"torch_tensorrt version: {trt.__version__}")
    return True


def load_dit_with_rmsnorm_fold(checkpoint_path):
    """Load WanModel from checkpoint with RMSNorm folding.

    Args:
        checkpoint_path: Path to .safetensors or .pth checkpoint.

    Returns:
        WanModel: Loaded and RMSNorm-folded model on CUDA in eval mode.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Create WanModel with standard FlashVSR config
    model = WanModel(
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

    # Handle 'model.' prefix in checkpoint keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Move to CUDA
    model = model.cuda()

    # Fold RMSNorm operations into weights
    print("Folding RMSNorm operations...")
    model = fold_dit_rmsnorms(model)

    return model


def make_trt_input_spec():
    """Create TensorRT input specification with dynamic shape ranges.

    Returns:
        trt.Input: Input spec for (B=1, T=1-64, C=16, H=128-2048, W=128-2048)
    """
    # Dynamic shape ranges: T:1-64, H/W:128-2048, B:1-8
    input_spec = trt.Input(
        (1, 1, 16, 128, 128),
        dtype=torch.float16,
        shape_ranges=[
            ((1, 1, 16, 128, 128), (4, 16, 16, 512, 512), (8, 64, 16, 2048, 2048))
        ],
    )
    return input_spec


def export_dit_for_trt(model, example_input):
    """Export DiT model using torch.export.

    Args:
        model: WanModel to export.
        example_input: Tuple of (latents, timesteps, contexts) tensors.

    Returns:
        Exported program from torch.export.
    """
    print("Exporting model via torch.export...")
    exported = torch.export.export(model, example_input)
    print("Export successful.")
    return exported


def create_trt_calibrator(dataloader, num_samples=320):
    """Create TensorRT calibrator using DataLoaderCalibrator.

    Args:
        dataloader: DataLoader providing calibration samples.
        num_samples: Number of samples to use for calibration.

    Returns:
        DataLoaderCalibrator configured for INT8 calibration.
    """
    calibrator = ptq.DataLoaderCalibrator(
        dataloader,
        calibrationAlgo=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
        num_samples=num_samples,
    )
    return calibrator


def compile_trt_engine(exported_dit, output_engine, calibrator, input_spec):
    """Compile TensorRT engine from exported model.

    Args:
        exported_dit: Exported model from torch.export.
        output_engine: Path to save the TensorRT engine.
        calibrator: Calibrator for INT8 quantization.
        input_spec: TensorRT input specification with shape ranges.
    """
    print(f"Compiling TensorRT engine to {output_engine}...")

    # Compile with INT8 precision and calibrator
    trt_model = trt.compile(
        exported_dit,
        inputs=[input_spec],
        enabled_precisions={torch.int8},
        ptq_calibrator=calibrator,
    )

    # Ensure output directory exists
    output_path = Path(output_engine)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save engine
    torch.jit.save(trt_model, output_engine)
    print(f"TensorRT engine saved to {output_engine}")


def main():
    """Main entry point for TensorRT INT8 compilation."""
    parser = argparse.ArgumentParser(
        description="Compile FlashVSR DiT to TensorRT INT8 engine"
    )
    parser.add_argument(
        "--input_ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (.safetensors or .pth)",
    )
    parser.add_argument(
        "--calibration_cache",
        type=str,
        default=None,
        help="Path to calibration cache (for reference, not used directly)",
    )
    parser.add_argument(
        "--output_engine",
        type=str,
        required=True,
        help="Path to save TensorRT engine",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=320,
        help="Number of calibration samples (default: 320)",
    )

    args = parser.parse_args()

    # Check torch_tensorrt availability
    if not check_torch_tensorrt():
        sys.exit(1)

    # Load model with RMSNorm folding
    model = load_dit_with_rmsnorm_fold(args.input_ckpt)

    # Create example input for torch.export
    # Shape: (B=1, T=1, C=16, H=24, W=24) - small example for export
    print("Creating example input...")
    latents = torch.randn(1, 1, 16, 24, 24, device="cuda", dtype=torch.float16)
    timesteps = torch.tensor([0], device="cuda", dtype=torch.int64)
    contexts = torch.randn(1, 10, 4096, device="cuda", dtype=torch.float16)
    example_input = (latents, timesteps, contexts)

    # Export model
    exported_dit = export_dit_for_trt(model, example_input)

    # Create calibrator from FlashVSRTQDataset DataLoader
    print(f"Creating calibrator with {args.num_samples} samples...")
    from scripts.ptq.calibrator_w8a8 import FlashVSRTQDataset
    from torch.utils.data import DataLoader

    calibration_dataset = FlashVSRTQDataset(
        root="datasets",
        num_samples=args.num_samples,
        frame_size=(24, 24),
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=1,
        num_workers=0,  # Single thread for simplicity
        shuffle=False,
    )

    calibrator = create_trt_calibrator(calibration_loader, num_samples=args.num_samples)

    # Create input spec
    input_spec = make_trt_input_spec()

    # Compile and save engine
    compile_trt_engine(exported_dit, args.output_engine, calibrator, input_spec)

    print("Compilation complete!")


if __name__ == "__main__":
    main()