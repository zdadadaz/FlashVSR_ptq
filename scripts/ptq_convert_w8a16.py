import argparse
import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.wan_video_dit import WanModel
from src.models.quantization.quant import convert_model_to_w8a16

def main():
    parser = argparse.ArgumentParser(description="Convert FlashVSR model to W8A16 format.")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to original .safetensors or .pth")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Path to save quantized model")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.input_ckpt}...")
    if args.input_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file, save_file
        state_dict = load_file(args.input_ckpt)
    else:
        state_dict = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)

    print("Instantiating generic FP16 model structure...")
    # These configs match the standard FlashVSR
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
        text_dim=4096
    )
    
    print("Loading weights...")
    model.load_state_dict(state_dict, strict=False)

    print("Converting to W8A16...")
    model = convert_model_to_w8a16(model)
    
    print(f"Saving quantized checkpoint to {args.output_ckpt}...")
    if args.output_ckpt.endswith('.safetensors'):
        save_file(model.state_dict(), args.output_ckpt)
    else:
        torch.save(model.state_dict(), args.output_ckpt)
    print("Done!")

if __name__ == "__main__":
    main()
