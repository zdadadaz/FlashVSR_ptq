import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.utils import Buffer_LQ4x_Proj


class LQProjONNXWrapper(nn.Module):
    """Wrapper for LQ_proj_in to export to ONNX."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video):
        # video: (B, C, F, H, W) = (1, 3, T, 16, 16) - typical LQ input after 4x downscale
        # Output: (layer_num, B, out_dim) = (30, 1, 1536)
        result = self.model(video)
        # result is a list of tensors, stack them
        return torch.stack(result, dim=0)


def export_lq_proj_onnx(ckpt_path, output_path, device="cpu", dummy_weights=False):
    print(f"Loading LQ_proj_in from {ckpt_path}...")

    # Create model with in_dim=3, out_dim=1536, layer_num=30 (matches ckpt)
    model = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=30)
    model.eval()

    if not dummy_weights:
        print(f"  Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys (sample): {missing[:5]}")
    else:
        print("  [dummy_weights] Skipping checkpoint load, model stays initialized.")

    wrapper = LQProjONNXWrapper(model)

    # Dummy input: (B, C, F, H, W) = (1, 3, 5, 16, 16) - need F >= 5 for multiple iterations
    # iter_ = 1 + (t-1) // 4, so t=5 -> iter_=2, t=4 -> iter_=1 (causes empty out_x)
    B, C, F, H, W = 1, 3, 5, 16, 16
    dummy_video = torch.randn(B, C, F, H, W).to(device)

    print(f"Exporting LQ_proj_in to {output_path}...")

    torch.onnx.export(
        wrapper,
        (dummy_video,),
        output_path,
        input_names=["video"],
        output_names=["features"],
        dynamic_axes={
            "video": {0: "batch", 2: "frames"},
            "features": {0: "layer_num", 1: "batch"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print("Export complete!")


def test_lq_proj_onnx(onnx_path, device="cpu"):
    import onnxruntime as ort
    print(f"Testing LQ_proj_in ONNX model at {onnx_path}...")

    providers = ['CPUExecutionProvider']
    if device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider']

    session = ort.InferenceSession(onnx_path, providers=providers)

    # Match export dimensions
    B, C, F, H, W = 1, 3, 5, 16, 16
    x = np.random.randn(B, C, F, H, W).astype(np.float32)

    print("Running inference...")
    outputs = session.run(None, {"video": x})
    print(f"Inference successful! Output shape: {outputs[0].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LQ_proj_in to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to LQ_proj_in.ckpt")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--test_only", action="store_true", help="Skip export, only test existing ONNX")
    parser.add_argument("--dummy_weights", action="store_true", help="Skip checkpoint load, export with initialized weights (produces single .onnx file)")
    args = parser.parse_args()

    if not args.test_only:
        export_lq_proj_onnx(args.ckpt, args.output, dummy_weights=args.dummy_weights)

    if os.path.exists(args.output):
        test_lq_proj_onnx(args.output)
        if not args.dummy_weights:
            output_dir = os.path.dirname(args.output) or "."
            for f in os.listdir(output_dir):
                if f.startswith("_") and not f.endswith(".onnx"):
                    try:
                        os.remove(os.path.join(output_dir, f))
                    except OSError:
                        pass
    else:
        print(f"Error: ONNX file {args.output} not found.")
