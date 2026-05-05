import argparse
import os
import sys
import torch
import time
import math
import cv2
import numpy as np
from pathlib import Path
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

def load_video_frames(video_path, num_frames=8, start_idx=0, max_size=256):
    """Load frames from video, resize if needed."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        frames.append(frame)
    cap.release()
    return frames

def calculate_psnr(img1, img2):
    """Calculate PSNR between two image tensors [0,1] range."""
    if img1.shape != img2.shape:
        # Resize to match
        if img1.ndim == 4:
            # (B, H, W, C)
            b, h, w, c = img2.shape
            img1_np = img1[:, :h, :w, :].cpu().numpy() if hasattr(img1, 'cpu') else img1
        else:
            img1_np = img1.cpu().numpy() if hasattr(img1, 'cpu') else img1
        img2_np = img2.cpu().numpy() if hasattr(img2, 'cpu') else img2
    else:
        img1_np = img1.cpu().numpy() if hasattr(img1, 'cpu') else img1
        img2_np = img2.cpu().numpy() if hasattr(img2, 'cpu') else img2

    mse = np.mean((img1_np.astype(np.float32) - img2_np.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_psnr_tensor(t1, t2):
    """PSNR between two tensors on same device."""
    if t1.shape != t2.shape:
        t2 = t2[:, :t1.shape[1], :t1.shape[2], :]
    mse = torch.mean((t1.float() - t2.float()) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def run_inference(pipe, frames_tensor, mode="tiny", seed=123):
    """Run flashvsr inference."""
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
        seed=seed,
        force_offload=False,
        enable_debug=False,
        chunk_size=0,
        resize_factor=1.0,
        mode=mode,
        context_pad=0
    )
    return output

def main():
    parser = argparse.ArgumentParser(description="Test FlashVSR quantization with real video data.")
    parser.add_argument("--model", type=str, default="FlashVSR", choices=["FlashVSR", "FlashVSR-v1.1"])
    parser.add_argument("--mode", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str,
        default="/home/user/apps/FlashVSRptq/FlashVSR_Integrated/datasets/test",
        help="Path to test dataset")
    parser.add_argument("--video", type=str, default=None,
        help="Specific video to use, or auto-select from dataset")
    parser.add_argument("--quant", type=str, default="W8A16",
        choices=["W8A16", "W8A8_SmoothQuant"],
        help="Quantization mode to test")
    args = parser.parse_args()

    # Find video
    if args.video and os.path.exists(args.video):
        video_path = args.video
    else:
        # Auto-select first available video from SPMCS or VideoLQ
        for dataset in ["SPMCS", "VideoLQ"]:
            lq_path = Path(args.dataset) / dataset / "LQ-Video"
            if lq_path.exists():
                videos = list(lq_path.glob("*.mkv")) + list(lq_path.glob("*.mp4"))
                if videos:
                    video_path = str(videos[0])
                    print(f"Auto-selected video: {video_path}")
                    break
        else:
            raise FileNotFoundError(f"No videos found in {args.dataset}")

    print(f"Loading video: {video_path}")
    frames = load_video_frames(video_path, num_frames=4, max_size=128)
    if len(frames) < 2:
        raise RuntimeError(f"Not enough frames loaded from {video_path}")

    # Convert to tensor (B, H, W, C) float32 [0,1]
    frames_np = np.stack(frames).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_np)
    print(f"Input frames shape: {frames_tensor.shape}")

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

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    baseline_output = run_inference(pipe_base, frames_tensor, mode=args.mode, seed=42)
    baseline_time = time.time() - start_time
    print(f"\nBaseline inference took: {baseline_time:.2f}s")

    if torch.cuda.is_available():
        peak_vram_base = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM (Baseline): {peak_vram_base:.2f} GB")

    del pipe_base
    torch.cuda.empty_cache()

    # --- Run W8A16 ---
    print("\n" + "="*50)
    print(f"--- Running {args.quant} Quantized Model ---")
    print("="*50)
    pipe_w8a16 = init_pipeline(
        model=args.model,
        mode=args.mode,
        device=args.device,
        dtype=dtype,
        vae_model="Wan2.1",
        quantize_mode=args.quant
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    w8a16_output = run_inference(pipe_w8a16, frames_tensor, mode=args.mode, seed=42)
    w8a16_time = time.time() - start_time
    print(f"\nW8A16 inference took: {w8a16_time:.2f}s")

    if torch.cuda.is_available():
        peak_vram_w8a16 = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM (W8A16): {peak_vram_w8a16:.2f} GB")
        if peak_vram_base > 0:
            print(f"VRAM Reduction: {peak_vram_base - peak_vram_w8a16:.2f} GB")

    del pipe_w8a16
    torch.cuda.empty_cache()

    # --- Calculate PSNR ---
    psnr_w8a16 = calculate_psnr_tensor(baseline_output, w8a16_output)
    print(f"\nDebug: baseline_output shape={baseline_output.shape}, device={baseline_output.device}")
    print(f"Debug: w8a16_output shape={w8a16_output.shape}, device={w8a16_output.device}")
    diff = baseline_output.float() - w8a16_output.float()
    print(f"Debug: diff min={diff.min().item()}, max={diff.max().item()}, mean={diff.abs().mean().item()}")
    mse = torch.mean(diff ** 2).item()
    print(f"Debug: mse={mse}")
    print("\n" + "="*50)
    print("--- Quality Metrics ---")
    print("="*50)
    print(f"PSNR ({args.quant} vs Baseline): {psnr_w8a16:.2f} dB")
    if psnr_w8a16 > 30:
        print("Excellent! PSNR > 30 dB indicates no perceptible quality loss.")
    elif psnr_w8a16 > 25:
        print("Good. PSNR 25-30 dB: minor quality degradation noticeable only in specific patterns.")
    else:
        print("Warning: PSNR is below 25 dB. Quality degradation is clearly visible.")

if __name__ == "__main__":
    main()