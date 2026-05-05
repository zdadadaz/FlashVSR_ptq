import argparse
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nodes import init_pipeline, flashvsr

def load_video_frames(video_path, num_frames=8, max_size=256):
    """Load frames from video, resize if needed."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        # Resize if too large
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        frames.append(frame)
    cap.release()
    return frames

def collect_activation_stats(model, frames_tensor, pipe, scale=2):
    """Run inference and collect stats for quantization calibration."""
    device = next(model.parameters()).device
    stats = {}

    # Hook to collect activations
    activation_collector = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation_collector[name] = output[0].detach()
            else:
                activation_collector[name] = output.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            h = module.register_forward_hook(hook_fn(name))
            hooks.append(h)

    # Run inference
    try:
        with torch.no_grad():
            output = flashvsr(
                pipe=pipe,
                frames=frames_tensor,
                scale=scale,
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
                seed=42,
                force_offload=False,
                enable_debug=False,
                chunk_size=0,
                resize_factor=1.0,
                mode="tiny",
                context_pad=0
            )
    except Exception as e:
        print(f"Warning: inference failed: {e}")

    # Remove hooks
    for h in hooks:
        h.remove()

    return activation_collector

def calibrate_quantization(model, pipe, dataset_path, num_videos=3, frames_per_video=4):
    """Calibrate model with real video data."""
    from src.models.quantization.quant import convert_model_to_w8a16

    # Find test videos
    video_dirs = [
        ("SPMCS", "LQ-Video"),
        ("VideoLQ", "LQ-Video"),
        ("RealVSR", "LQ-Video"),
    ]

    videos = []
    for dataset, subdir in video_dirs:
        path = Path(dataset_path) / dataset / subdir
        if path.exists():
            for f in sorted(path.glob("*.mkv"))[:2]:
                videos.append(str(f))
            for f in sorted(path.glob("*.mp4"))[:2]:
                videos.append(str(f))
        if len(videos) >= num_videos:
            break

    if not videos:
        print(f"No videos found in {dataset_path}")
        return None

    print(f"Using {len(videos)} videos for calibration")
    device = next(model.parameters()).device

    all_stats = {}

    for i, video_path in enumerate(videos[:num_videos]):
        print(f"Processing video {i+1}/{len(videos)}: {video_path}")
        frames = load_video_frames(video_path, num_frames=frames_per_video)
        if len(frames) < 2:
            continue

        # Convert to tensor (B, H, W, C) -> (B, H, W, 3) float32 [0,1]
        frames_np = np.stack(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)

        try:
            stats = collect_activation_stats(model, frames_tensor, pipe)
            # Merge stats
            for name, act in stats.items():
                if name not in all_stats:
                    all_stats[name] = []
                all_stats[name].append(act)
        except Exception as e:
            print(f"Failed on {video_path}: {e}")
            continue

    return all_stats

def compute_quantization_scales(all_stats):
    """Compute per-channel scales based on activation stats."""
    scales = {}
    for name, acts in all_stats.items():
        if not acts:
            continue
        acts_cat = torch.cat([a.flatten() for a in acts])
        # Per-channel scale based on max absolute value
        # For simplicity, use the overall scale per layer
        scale = torch.max(torch.abs(acts_cat)) / 127.0
        scales[name] = scale.clamp(min=1e-8)
    return scales

def main():
    parser = argparse.ArgumentParser(description="Calibrate FlashVSR W8A16 quantization with real data.")
    parser.add_argument("--dataset", type=str,
        default="/home/user/apps/FlashVSRptq/FlashVSR_Integrated/datasets/test",
        help="Path to test dataset")
    parser.add_argument("--output", type=str,
        default="quant_scales.pt",
        help="Path to save quantization scales")
    parser.add_argument("--num_videos", type=int, default=3, help="Number of videos to use")
    parser.add_argument("--frames", type=int, default=4, help="Frames per video")
    parser.add_argument("--model", type=str, default="FlashVSR", choices=["FlashVSR", "FlashVSR-v1.1"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    dtype = torch.bfloat16
    print(f"Initializing pipeline for calibration...")

    pipe = init_pipeline(
        model=args.model,
        mode="tiny",
        device=args.device,
        dtype=dtype,
        vae_model="Wan2.1",
        quantize_mode="None"
    )

    model = pipe.denoising_model()
    model.cuda()

    print(f"Running calibration with {args.num_videos} videos, {args.frames} frames each...")
    all_stats = calibrate_quantization(
        model, pipe, args.dataset,
        num_videos=args.num_videos,
        frames_per_video=args.frames
    )

    if all_stats:
        scales = compute_quantization_scales(all_stats)
        torch.save(scales, args.output)
        print(f"Saved calibration scales to {args.output}")
    else:
        print("No valid activation stats collected!")

    # Clean up
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()