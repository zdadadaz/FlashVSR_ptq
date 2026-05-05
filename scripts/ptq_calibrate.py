import argparse
import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.wan_video_dit import WanModel
from src.models.quantization.smoothquant import inject_observers, calculate_smoothquant_scales

def main():
    parser = argparse.ArgumentParser(description="Calibrate FlashVSR model for W8A8 SmoothQuant.")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to original .safetensors or .pth")
    parser.add_argument("--output_scales", type=str, required=True, help="Path to save quantization scales .pt")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.input_ckpt}...")
    if args.input_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(args.input_ckpt)
    else:
        state_dict = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)

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
    
    model.load_state_dict(state_dict, strict=False)

    print("Injecting Observers...")
    model = inject_observers(model)
    model.cuda()
    
    print("Running fake calibration pass (replace with real video frames)...")
    # Simulate a calibration pass with random data
    # In reality, this should invoke the full pipeline with a dataset
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(1, 10, 16, 24, 24, device='cuda')
            t = torch.randn(1, device='cuda')
            context = torch.randn(1, 10, 4096, device='cuda')
            seq_len = 10*24*24
            # model(x, t, context, seq_len) # pseudo forward pass
            
    print("Calculating SmoothQuant scales...")
    scales = calculate_smoothquant_scales(model, alpha=0.5)
    
    print(f"Saving scales to {args.output_scales}...")
    torch.save(scales, args.output_scales)
    print("Done!")

if __name__ == "__main__":
    main()
