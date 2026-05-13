import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.TCDecoder import TAEW2_1DiffusersWrapper, build_tcdecoder

class TCDecoderONNXWrapper(nn.Module):
    """Wrapper for TAEHV to export to ONNX with simple decode interface."""
    def __init__(self, taehv_decoder):
        super().__init__()
        self.decoder = taehv_decoder

    def forward(self, x):
        # x: (N, C, T, H, W) - latents in NCHW format
        # decode_video expects (n, t, c, h, w) so transpose
        x_t = x.transpose(1, 2)
        # decode_video with parallel=True for ONNX export
        result = self.decoder.decode_video(x_t, parallel=True)
        # result is (n, t, 3, h, w) - transpose back to (n, 3, t, h, w)
        return result.transpose(1, 2)


def export_tcdecoder_onnx(ckpt_path, output_path, device="cpu", dummy_weights=False):
    print(f"Loading TCDecoder from {ckpt_path}...")

    from src.models.TCDecoder import TAEHV
    # TCDecoder.ckpt was trained with channels=[512, 256, 128, 128]
    decoder = TAEHV(checkpoint_path=None, channels=[512, 256, 128, 128])
    decoder.eval()

    if not dummy_weights:
        print(f"  Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        new_ckpt = {}
        for k, v in ckpt.items():
            new_ckpt[k] = v
        missing, unexpected = decoder.load_state_dict(new_ckpt, strict=False)
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys (sample): {missing[:5]}")
    else:
        print("  [dummy_weights] Skipping checkpoint load, model stays initialized.")

    # Create wrapper
    wrapper = TCDecoderONNXWrapper(decoder)

    # Dummy input: (N, C, T, H, W) = (1, 16, 4, 64, 64) - typical latent size
    # N=batch=1, C=latent_channels=16, T=frames, H/W=spatial dims
    N, C, T, H, W = 1, 16, 4, 64, 64
    dummy_x = torch.randn(N, C, T, H, W).to(device)

    print(f"Exporting TCDecoder to {output_path}...")

    torch.onnx.export(
        wrapper,
        (dummy_x,),
        output_path,
        input_names=["latents"],
        output_names=["video"],
        dynamic_axes={
            "latents": {0: "batch", 2: "frames", 3: "height", 4: "width"},
            "video": {0: "batch", 2: "frames", 3: "height", 4: "width"}
        },
        opset_version=17,
        do_constant_folding=False  # Avoid potential issues with dynamic shapes
    )
    print("Export complete!")


def test_tcdecoder_onnx(onnx_path, device="cpu"):
    import onnxruntime as ort
    print(f"Testing TCDecoder ONNX model at {onnx_path}...")

    providers = ['CPUExecutionProvider']
    if device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider']

    session = ort.InferenceSession(onnx_path, providers=providers)

    # Match export dimensions
    N, C, T, H, W = 1, 16, 4, 64, 64
    x = np.random.randn(N, C, T, H, W).astype(np.float32)

    print("Running inference...")
    outputs = session.run(None, {"latents": x})
    print(f"Inference successful! Output shape: {outputs[0].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TCDecoder to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to TCDecoder.ckpt")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--test_only", action="store_true", help="Skip export, only test existing ONNX")
    parser.add_argument("--dummy_weights", action="store_true", help="Skip checkpoint load, export with initialized weights (produces single .onnx file without external data)")
    args = parser.parse_args()

    if not args.test_only:
        export_tcdecoder_onnx(args.ckpt, args.output, dummy_weights=args.dummy_weights)

    if os.path.exists(args.output):
        test_tcdecoder_onnx(args.output)
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