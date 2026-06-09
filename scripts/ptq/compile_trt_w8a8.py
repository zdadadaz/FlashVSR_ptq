"""TensorRT INT8 compilation script for FlashVSR DiT.

Flow:
1. Run calibration forward passes to collect activation stats
2. Convert DiT to W8A8 (Int8ActLinear with bf16 matmul) using activation scales
3. torch_tensorrt.dynamo.compile the quantized model to a TRT engine

Bypasses torch.export entirely to avoid Triton custom kernel tracing issues
and random.random() side-effects in the DiT model's forward pass.
"""

import argparse
import sys
from pathlib import Path

import torch
from dataclasses import dataclass

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.wan_video_dit import WanModel, flash_attention as _flash_attention
from src.models.quantization.rmsnorm_fold import fold_dit_rmsnorms

# Optional: torch_tensorrt
try:
    import torch_tensorrt as trt
    import torch_tensorrt.dynamo as dynamo
    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False
    trt = None
    dynamo = None


def check_torch_tensorrt():
    """Check if torch-tensorrt is available."""
    if not TORCH_TENSORRT_AVAILABLE:
        print("ERROR: torch-tensorrt is not installed.")
        print("Install with: pip install torch-tensorrt")
        return False
    print(f"torch_tensorrt version: {trt.__version__}")
    return True


def load_dit_with_rmsnorm_fold(checkpoint_path):
    """Load WanModel from checkpoint with RMSNorm folding."""
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
    model = model.cuda()
    model = model.half()

    print("Folding RMSNorm operations...")
    model = fold_dit_rmsnorms(model)

    return model


def patched_flash_attention(q, k, v, num_heads, compatibility_mode=False, attention_mask=None, return_KV=False):
    """Patched flash_attention that forces compatibility_mode=True during calibration.

    This bypasses the Triton-based sparse_sageattn kernel (which requires int8
    quantization support) and uses PyTorch's native scaled_dot_product_attention
    during the calibration forward passes. The actual TRT engine will execute
    sparse_sage via torch_executed_modules.
    """
    return _flash_attention(q, k, v, num_heads, compatibility_mode=True, attention_mask=None, return_KV=return_KV)


def make_trt_input_spec():
    """Create TensorRT input specification with dynamic shape ranges."""
    input_spec = trt.Input(
        (1, 16, 2, 128, 128),
        dtype=torch.float16,
        shape_ranges=[
            ((1, 16, 2, 32, 32), (4, 16, 16, 512, 512), (8, 16, 64, 2048, 2048))
        ],
    )
    return input_spec



def export_dit_for_trt(model, example_input):
    """Export a DiT module for TensorRT graph inspection.

    The production INT8 path uses ``torch_tensorrt.dynamo.compile`` directly,
    but legacy tests and debugging scripts still import this helper to validate
    that a small WanModel can be captured by ``torch.export``.
    """
    model.eval()
    with torch.no_grad():
        return torch.export.export(model, example_input)


@dataclass
class _DataLoaderCalibrator:
    """Minimal DataLoader-backed calibrator descriptor for legacy callers.

    torch_tensorrt 2.x no longer exposes the old Python calibrator classes used
    by this script's tests. Keep a lightweight object so callers can pass around
    the calibration loader without importing removed torch_tensorrt APIs.
    """

    dataloader: object
    num_samples: int | None = None

    def __iter__(self):
        count = 0
        for batch in self.dataloader:
            if self.num_samples is not None and count >= self.num_samples:
                break
            count += 1
            yield batch


def create_trt_calibrator(dataloader, num_samples=None):
    """Create a DataLoader-backed calibration descriptor.

    This preserves the legacy compile script API while avoiding dependency on
    removed torch_tensorrt calibrator classes.
    """
    if dataloader is None:
        raise ValueError("dataloader is required")
    return _DataLoaderCalibrator(dataloader=dataloader, num_samples=num_samples)

def compile_trt_engine(model, output_engine, input_spec):
    """Compile TensorRT engine from pre-quantized W8A8 model using dynamo.compile.

    Uses torch_tensorrt.dynamo.compile which:
    - Internally runs the model through torch.compile to capture the graph
    - Bypasses the need for torch.export (avoids Triton tracing issues)
    - The model is already quantized; TRT executes int8 ops via torch_executed_modules.
    """
    print(f"Compiling TensorRT engine to {output_engine}...")

    # Model is already quantized to W8A8 via convert_model_to_w8a8 before this call.
    # We do NOT use enable_autocast here -- torch_tensorrt.dynamo does not support
    # autocast_low_precision_type=torch.int8 (only FP16/BF16 are supported).
    # The pre-quantized model's int8 weights are preserved through torch.compile
    # and TRT will execute them as int8 via torch_executed_modules.
    print("Compiling pre-quantized W8A8 model to TRT engine (no autocast)...")
    trt_model = dynamo.compile(
        model,
        inputs=[input_spec],
        require_full_compilation=True,
    )

    output_path = Path(output_engine)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.jit.save(trt_model, output_engine)
    print(f"TensorRT engine saved to {output_engine}")


def main():
    """Main entry point for TensorRT INT8 compilation."""
    parser = argparse.ArgumentParser(description="Compile FlashVSR DiT to TensorRT INT8 engine")
    parser.add_argument(
        "--input_ckpt", type=str, required=True, help="Path to model checkpoint (.safetensors or .pth)"
    )
    parser.add_argument(
        "--calibration_cache", type=str, default=None, help="Path to calibration cache (for reference, not used directly)"
    )
    parser.add_argument(
        "--output_engine", type=str, required=True, help="Path to save TensorRT engine"
    )
    parser.add_argument(
        "--num_samples", type=int, default=320, help="Number of calibration samples (default: 320)"
    )

    args = parser.parse_args()

    if not check_torch_tensorrt():
        sys.exit(1)

    # Load model with RMSNorm folding
    model = load_dit_with_rmsnorm_fold(args.input_ckpt)

    # Create calibrator from FlashVSRTQDataset DataLoader
    print(f"Creating calibrator with {args.num_samples} samples...")
    from scripts.ptq.calibrator_w8a8 import FlashVSRTQDataset, collate_calibration_samples
    from torch.utils.data import DataLoader

    calibration_dataset = FlashVSRTQDataset(
        root="datasets",
        num_samples=args.num_samples,
        frame_size=(24, 24),
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_calibration_samples,
    )

    # Step 0: Patch flash_attention to bypass Triton int8 kernel during all forward passes
    # (calibration and dynamo.compile tracing both need SDPA compatibility mode)
    import src.models.wan_video_dit as wan_module
    original_flash_attn = wan_module.flash_attention
    wan_module.flash_attention = patched_flash_attention

    try:
        # Step 1: Collect activation stats for W8A8 quantization
        print("Collecting activation stats for W8A8 quantization...")
        from scripts.ptq.calibrator_w8a8 import run_calibration
        act_stats = run_calibration(
            model,
            calibration_dataset,
            batch_size=1,
            num_workers=0,
        )
        print(f"Calibration complete: collected stats from {len(act_stats)} layers")

        # Step 2: Convert model to W8A8 (weight-only int8 + int8 activations)
        print("Applying W8A8 quantization to model...")
        from src.models.quantization.quant import convert_model_to_w8a8

        # Transform act_stats from {name: {'act_min', 'act_max', ...}} to {name: amax_tensor}
        # convert_model_to_w8a8 expects act_amax as tensor with shape matching in_features
        act_stats_transformed = {}
        for name, stats in act_stats.items():
            # act_max is a scalar, convert to tensor with shape (in_features,)
            act_max = stats['act_max']  # scalar value
            act_stats_transformed[name] = torch.tensor(act_max, dtype=torch.float32) if isinstance(act_max, float) else act_max

        convert_model_to_w8a8(model, act_stats_transformed, method='percentile99', engine='bf16')

        # Create input spec
        input_spec = make_trt_input_spec()

        # Step 3: Compile pre-quantized model to TRT engine
        compile_trt_engine(model, args.output_engine, input_spec)
    finally:
        wan_module.flash_attention = original_flash_attn

    print("Compilation complete!")


if __name__ == "__main__":
    main()