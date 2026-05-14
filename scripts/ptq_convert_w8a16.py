import argparse
import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.wan_video_dit import WanModel
from src.models.quantization.quant import convert_model_to_w8a16, convert_model_to_w8a8

def main():
    parser = argparse.ArgumentParser(description="Convert FlashVSR model to W8A16 or W8A8 format.")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to original .safetensors or .pth")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Path to save quantized model")
    parser.add_argument("--quantize_mode", type=str, default="w8a16",
                        choices=["w8a16", "w8a8"],
                        help="Quantization mode: w8a16 (weight-only int8) or w8a8 (int8 weights + activations)")
    parser.add_argument("--w8a8_engine", type=str, default="bf16",
                        choices=["bf16", "int8mm"],
                        help="W8A8 matmul engine: bf16 (higher quality) or int8mm (faster, experimental)")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.input_ckpt}...")
    if args.input_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file, save_file
        state_dict = load_file(args.input_ckpt)
    else:
        state_dict = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)

    print("Instantiating generic FP16 model structure...")
    # These configs match the standard FlashVSR-v1.1
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
        text_dim=4096
    )
    
    print("Loading weights...")
    # Handle state_dict keys if they have 'model.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)

    if args.quantize_mode == "w8a16":
        print("Converting to W8A16 (weight-only int8)...")
        model = convert_model_to_w8a16(model)
    elif args.quantize_mode == "w8a8":
        print(f"Converting to W8A8 (int8 weights + activations, engine={args.w8a8_engine})...")
        # act_stats=None will fallback to WeightOnlyInt8Linear for layers without activation stats
        model = convert_model_to_w8a8(model, act_stats=None, engine=args.w8a8_engine)

    print(f"Saving quantized checkpoint to {args.output_ckpt}...")
    if args.output_ckpt.endswith('.safetensors'):
        save_file(model.state_dict(), args.output_ckpt)
    else:
        torch.save(model.state_dict(), args.output_ckpt)
    print("Done!")

if __name__ == "__main__":
    main()
