import argparse
import os
import sys
import torch
import time
import math
from unittest.mock import MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Mock ComfyUI modules for standalone CLI operation
folder_paths_mock = MagicMock()
folder_paths_mock.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
folder_paths_mock.get_filename_list = MagicMock(return_value=[])
sys.modules['folder_paths'] = folder_paths_mock

comfy_mock = MagicMock()
comfy_utils_mock = MagicMock()
comfy_utils_mock.ProgressBar = MagicMock()
sys.modules['comfy'] = comfy_mock
sys.modules['comfy.utils'] = comfy_utils_mock

from nodes import init_pipeline, flashvsr

def calculate_psnr(img1, img2):
    # img1 and img2 are tensors, typically in [0, 1] range float if normalized
    if img1.dtype != img2.dtype:
        img2 = img2.to(img1.dtype)
    mse = torch.mean((img1.float() - img2.float()) ** 2).item()
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0 # Or 255 if images are uint8
    if torch.max(img1) > 2.0:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def run_inference(pipe, frames_tensor, mode="tiny"):
    # Run flashvsr with default testing params
    output = flashvsr(
        pipe=pipe,
        frames=frames_tensor,
        scale=2.0,
        color_fix=True,
        color_fix_method="wavelet",
        tiled_vae=False,
        tiled_dit=False,
        tile_size=256,
        tile_overlap=16,
        unload_dit=False,
        sparse_ratio=0.5,
        kv_ratio=0.5,
        local_range=128,
        seed=123,
        force_offload=False,
        enable_debug=False,
        chunk_size=0,
        resize_factor=1.0,
        mode=mode,
        context_pad=0
    )
    return output

def main():
    parser = argparse.ArgumentParser(description="Test FlashVSR quantization.")
    parser.add_argument("--model", type=str, default="FlashVSR", choices=["FlashVSR", "FlashVSR-v1.1"])
    parser.add_argument("--mode", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print("Generating dummy input frames: shape=[2, 64, 64, 3]...")
    torch.manual_seed(42)
    # Using 64x64 to avoid OOM or long loading times during tests
    # Frames expected to be float32 in [0, 1]
    dummy_frames = torch.rand((2, 64, 64, 3), dtype=torch.float32)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    # --- Run FP16/BF16 Baseline ---
    print("\n" + "="*50)
    print("--- Running Baseline (No Quantization) ---")
    print("="*50)
    pipe_base = init_pipeline(
        model=args.model,
        mode=args.mode,
        device=args.device,
        dtype=dtype,
        vae_model="Wan2.1",
        quantize_mode="None"
    )
    
    start_time = time.time()
    baseline_output = run_inference(pipe_base, dummy_frames, mode=args.mode)
    baseline_time = time.time() - start_time
    print(f"\nBaseline inference took: {baseline_time:.2f}s")
    
    if torch.cuda.is_available():
        peak_vram_base = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM (Baseline): {peak_vram_base:.2f} GB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    # Free memory
    del pipe_base
    torch.cuda.empty_cache()

    # --- Run W8A16 ---
    print("\n" + "="*50)
    print("--- Running W8A16 Quantized Model ---")
    print("="*50)
    pipe_w8a16 = init_pipeline(
        model=args.model,
        mode=args.mode,
        device=args.device,
        dtype=dtype,
        vae_model="Wan2.1",
        quantize_mode="W8A16"
    )
    
    start_time = time.time()
    w8a16_output = run_inference(pipe_w8a16, dummy_frames, mode=args.mode)
    w8a16_time = time.time() - start_time
    print(f"\nW8A16 inference took: {w8a16_time:.2f}s")
    
    if torch.cuda.is_available():
        peak_vram_w8a16 = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM (W8A16): {peak_vram_w8a16:.2f} GB")
        print(f"VRAM Reduction: {peak_vram_base - peak_vram_w8a16:.2f} GB")
        
    del pipe_w8a16
    torch.cuda.empty_cache()
    
    # --- Calculate PSNR ---
    psnr_w8a16 = calculate_psnr(baseline_output, w8a16_output)
    print("\n" + "="*50)
    print("--- Quality Metrics ---")
    print("="*50)
    print(f"PSNR (W8A16 vs Baseline): {psnr_w8a16:.2f} dB")
    if psnr_w8a16 > 30:
        print("Excellent! PSNR > 30 dB indicates no perceptible quality loss.")
    else:
        print("Warning: PSNR is below 30 dB. Quality might be visibly degraded.")

if __name__ == "__main__":
    main()
