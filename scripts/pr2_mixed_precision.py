#!/usr/bin/env python3
"""
PR #2: Mixed-Precision A16 Strategy for FlashVSR W8A8 PTQ

Goal: Use W8A16 (weight-only INT8) for sensitive layers, W8A8 SmoothQuant for robust layers.

Background:
- W8A16 all layers: 36.68 dB (from PR #0)
- W8A8 all layers (SmoothQuant): ~13 dB (from PR #1)
- Gap = ~23 dB — from activation quantization error

PR #0 per-group ablation ranking (sensitivity, lower = more sensitive):
  1. FFN: 38.24 dB ← MOST SENSITIVE → A16
  2. Embed: 38.68 dB → A16
  3. SelfAttn.V: 39.04 dB
  4. CrossAttn.V: 39.28 dB
  5. SelfAttn.Q: 39.30 dB
  6. CrossAttn.O: 39.30 dB
  7. SelfAttn.K: 39.39 dB
  8. Other (LQ_proj_in): 39.56 dB → A16
  9. CrossAttn.K: 39.60 dB
  10. SelfAttn.O: 39.69 dB
  11. CrossAttn.Q: 39.72 dB
  12. Head: 43.79 dB ← MOST ROBUST → A8

Strategy:
- A16: FFN (60), Embed (5), LQ_proj_in (1) = 66 layers → W8A16
- A8:  All attention QKV/O (300), Head (1) = 301 layers → W8A8 SmoothQuant

Expected: W8A8 for 300 robust layers + W8A16 for 66 sensitive layers
          → PSNR close to W8A16 baseline (36.68 dB)

Author: localpc
Date: 2026-05-06
"""

import argparse
import os
import sys
import time
import math
import json

# ── Setup ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from unittest.mock import MagicMock
folder_paths_mock = MagicMock()
folder_paths_mock.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
sys.modules['folder_paths'] = folder_paths_mock
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()

from nodes import init_pipeline, flashvsr
from src.models.quantization.smoothquant import inject_observers
from src.models.quantization.quant import (
    WeightOnlyInt8Linear,
    SmoothQuantLinear,
    convert_model_to_w8a8_smoothquant,
)


# ── Helper: Load video frames ────────────────────────────────────────────────
def load_video_frames(video_path, num_frames=4, max_size=128):
    """Load frames from a video file."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    read_count = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            scale = max_size / max(h, w)
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h),
                                      interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
            read_count += 1
            if read_count >= num_frames:
                break

    cap.release()
    return frames


# ── Conversion function with A16 exclude list ────────────────────────────────
def convert_model_mixed_precision(model, act_stats, alpha=0.5, method='max',
                                  a16_exclude_patterns=None, device='cuda:0'):
    """
    Replace nn.Linear with SmoothQuantLinear (W8A8), except for modules whose
    names match any pattern in a16_exclude_patterns (those get W8A16).

    a16_exclude_patterns: list of substrings; if any matches module name → W8A16
    """
    if a16_exclude_patterns is None:
        a16_exclude_patterns = []

    converted_sq = 0
    converted_w8a16 = 0
    fallback_count = 0

    # Pre-move act_stats to target device
    act_stats_dev = {}
    for k, v in act_stats.items():
        if isinstance(v, torch.Tensor):
            act_stats_dev[k] = v.to(device)
        else:
            act_stats_dev[k] = v

    def convert_module(mod, prefix=''):
        nonlocal converted_sq, converted_w8a16, fallback_count
        for name, module in mod.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            is_linear = isinstance(module, torch.nn.Linear)
            is_observer = hasattr(module, 'act_amax') and hasattr(module, 'weight')

            if is_linear or is_observer:
                weight_data = module.weight.data if hasattr(module, 'weight') else None
                bias_data = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None

                # Check if this module should use A16
                use_a16 = any(pat in full_name for pat in a16_exclude_patterns)

                if use_a16 and weight_data is not None:
                    # W8A16 for this layer
                    w8a16 = WeightOnlyInt8Linear.from_float(module, method=method)
                    setattr(mod, name, w8a16)
                    converted_w8a16 += 1
                elif weight_data is not None:
                    # Try SmoothQuant (W8A8)
                    act_amax = act_stats_dev.get(full_name)
                    if act_amax is None:
                        act_amax = act_stats_dev.get(name)

                    if act_amax is not None:
                        expected_in = module.in_features
                        actual_act = act_amax.shape[0]
                        if actual_act == expected_in:
                            weight_amax_per_input = torch.amax(
                                torch.abs(weight_data.to(device)), dim=0)
                            scale = (torch.pow(act_amax.clamp(min=1e-8), alpha) /
                                     torch.pow(weight_amax_per_input.clamp(min=1e-8), 1.0 - alpha))
                            scale = torch.clamp(scale, min=1e-5, max=1e5)

                            class LinearWrapper:
                                def __init__(self, weight, bias, in_f, out_f):
                                    self.weight = torch.nn.Parameter(weight)
                                    self.bias = torch.nn.Parameter(bias) if bias is not None else None
                                    self.in_features = in_f
                                    self.out_features = out_f

                            wrapper = LinearWrapper(
                                weight_data, bias_data,
                                module.in_features, module.out_features)
                            sq_linear = SmoothQuantLinear.from_linear(
                                wrapper, act_amax, scale, weight_quant_max=124)
                            setattr(mod, name, sq_linear)
                            converted_sq += 1
                        else:
                            if is_linear:
                                setattr(mod, name, WeightOnlyInt8Linear.from_float(module, method=method))
                                fallback_count += 1
                    else:
                        if is_linear:
                            setattr(mod, name, WeightOnlyInt8Linear.from_float(module, method=method))
                            fallback_count += 1
                else:
                    if is_linear:
                        setattr(mod, name, WeightOnlyInt8Linear.from_float(module, method=method))
                        fallback_count += 1
            else:
                convert_module(module, full_name)

    convert_module(model)
    print(f"Mixed-precision conversion: {converted_sq} W8A8 SmoothQuant, "
          f"{converted_w8a16} W8A16, {fallback_count} fallback")
    return model


# ── A16 exclude configurations ──────────────────────────────────────────────
# Based on PR #0 per-group ablation: sensitivity from most to least sensitive
# FFN > Embed > SelfAttn.V > CrossAttn.V > ... > Head

A16_CONFIGS = {
    # Config A: FFN only (most sensitive group)
    "FFN_only": {
        "patterns": [".blocks.*.ffn."],
        "description": "FFN only → A16 (60 layers)"
    },

    # Config B: FFN + Embed
    "FFN_Embed": {
        "patterns": [".blocks.*.ffn.", ".text_embedding.", ".time_embedding.",
                     ".time_projection."],
        "description": "FFN + Embed → A16 (65 layers)"
    },

    # Config C: FFN + Embed + Other (LQ_proj_in)
    "FFN_Embed_Other": {
        "patterns": [".blocks.*.ffn.", ".text_embedding.", ".time_embedding.",
                     ".time_projection.", "LQ_proj_in"],
        "description": "FFN + Embed + LQ_proj_in → A16 (66 layers)"
    },

    # Config D: Most sensitive 4 groups
    "top4": {
        "patterns": [".blocks.*.ffn.", ".text_embedding.", ".time_embedding.",
                     ".time_projection.",
                     ".blocks.*.self_attn.v", ".blocks.*.cross_attn.v"],
        "description": "FFN + Embed + V (all) → A16 (126 layers)"
    },

    # Config E: All QKV attention (sensitive but robust-ish)
    "attention_QKV": {
        "patterns": [".blocks.*.ffn.", ".text_embedding.", ".time_embedding.",
                     ".time_projection.",
                     ".blocks.*.self_attn.q", ".blocks.*.self_attn.k",
                     ".blocks.*.self_attn.v",
                     ".blocks.*.cross_attn.q", ".blocks.*.cross_attn.k",
                     ".blocks.*.cross_attn.v",
                     "LQ_proj_in"],
        "description": "FFN + Embed + All QKV → A16 (246 layers)"
    },

    # Config F: Everything except Head (very aggressive A16)
    "all_except_Head": {
        "patterns": [".blocks.", ".text_embedding.", ".time_embedding.",
                     ".time_projection.", "LQ_proj_in"],
        "description": "Everything except Head → A16 (306 layers)"
    },
}


# ── Main test ────────────────────────────────────────────────────────────────
def test_mixed_precision_configs(video_path, num_frames=4, max_size=128,
                                  seed=42, dtype=torch.bfloat16):
    """Test different A16 exclude configurations."""

    # FP16 baseline
    print("\n" + "="*60)
    print("FP16 Baseline")
    print("="*60)
    torch.manual_seed(seed)
    pipe_fp16 = init_pipeline(model="FlashVSR", mode="tiny", device="cuda:0",
                              dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    frames = load_video_frames(video_path, num_frames=num_frames, max_size=max_size)
    frames_np = np.stack(frames).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_np).cuda()

    t0 = time.time()
    baseline_out = flashvsr(pipe=pipe_fp16, frames=frames_tensor, scale=2.0,
                              color_fix=True, color_fix_method="wavelet",
                              tiled_vae=False, tiled_dit=False, tile_size=256,
                              tile_overlap=16, unload_dit=False, sparse_ratio=0.5,
                              kv_ratio=0.5, local_range=128, seed=seed,
                              force_offload=False, enable_debug=False,
                              chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
    baseline_time = time.time() - t0
    print(f"FP16: {baseline_time:.2f}s")
    del pipe_fp16
    torch.cuda.empty_cache()

    # Collect activation stats for SmoothQuant layers
    print("\n" + "="*60)
    print("Collecting activation stats for W8A8 layers")
    print("="*60)
    pipe_collect = init_pipeline(model="FlashVSR", mode="tiny", device="cuda:0",
                                 dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    dit_collect = pipe_collect.denoising_model()
    dit_collect = inject_observers(dit_collect)
    if hasattr(pipe_collect, 'init_cross_kv'):
        pipe_collect.init_cross_kv(context_tensor=torch.randn(1, 10, 4096, device='cuda:0'))

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "datasets", "test")
    calibration_videos = []
    for subdir in ["SPMCS/LQ-Video", "VideoLQ/LQ-Video", "RealVSR/LQ-Video"]:
        vid_dir = os.path.join(dataset_path, subdir)
        if os.path.isdir(vid_dir):
            for fn in sorted(os.listdir(vid_dir))[:3]:
                calibration_videos.append(os.path.join(vid_dir, fn))

    print(f"Calibration videos: {len(calibration_videos)}")
    for vid_path in calibration_videos:
        vid_frames = load_video_frames(vid_path, num_frames=num_frames, max_size=max_size)
        if len(vid_frames) == 0:
            continue
        vid_np = np.stack(vid_frames).astype(np.float32) / 255.0
        vid_tensor = torch.from_numpy(vid_np).cuda()
        try:
            with torch.no_grad():
                _ = flashvsr(pipe=pipe_collect, frames=vid_tensor, scale=2.0,
                             color_fix=True, color_fix_method="wavelet",
                             tiled_vae=False, tiled_dit=False, tile_size=256,
                             tile_overlap=16, unload_dit=False, sparse_ratio=0.5,
                             kv_ratio=0.5, local_range=128, seed=seed,
                             force_offload=False, enable_debug=False,
                             chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
        except:
            pass

    # Extract act_stats
    act_stats = {}
    for name, module in dit_collect.named_modules():
        if hasattr(module, 'act_amax'):
            act_stats[name] = module.act_amax.clone().cuda()

    del pipe_collect, dit_collect
    torch.cuda.empty_cache()

    results = {}

    # W8A16 all (baseline comparison)
    print("\n" + "="*60)
    print("W8A16 All (reference for A16 strategy)")
    print("="*60)
    pipe_w8a16 = init_pipeline(model="FlashVSR", mode="tiny", device="cuda:0",
                               dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    dit_w8a16 = pipe_w8a16.denoising_model()
    from src.models.quantization.quant import convert_model_to_w8a16
    dit_w8a16 = convert_model_to_w8a16(dit_w8a16, method='max')

    t0 = time.time()
    out_w8a16 = flashvsr(pipe=pipe_w8a16, frames=frames_tensor, scale=2.0,
                           color_fix=True, color_fix_method="wavelet",
                           tiled_vae=False, tiled_dit=False, tile_size=256,
                           tile_overlap=16, unload_dit=False, sparse_ratio=0.5,
                           kv_ratio=0.5, local_range=128, seed=seed,
                           force_offload=False, enable_debug=False,
                           chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
    time_w8a16 = time.time() - t0
    psnr_w8a16 = (20 * math.log10(1.0 / math.sqrt(
        torch.mean((baseline_out - out_w8a16).float() ** 2).item()))
        if not torch.isnan(out_w8a16).any() else float('nan'))
    print(f"W8A16 All PSNR: {psnr_w8a16:.2f}dB, Time: {time_w8a16:.2f}s")
    results['W8A16_all'] = {'psnr': psnr_w8a16, 'time': time_w8a16}
    del pipe_w8a16, dit_w8a16
    torch.cuda.empty_cache()

    # Test each A16 config
    for config_name, config in A16_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name} — {config['description']}")
        print(f"A16 patterns: {config['patterns']}")
        print("="*60)

        pipe = init_pipeline(model="FlashVSR", mode="tiny", device="cuda:0",
                            dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
        dit = pipe.denoising_model()
        dit = dit.cuda()  # ensure model is on CUDA before conversion
        dit = convert_model_mixed_precision(
            dit, act_stats, alpha=0.5, method='max',
            a16_exclude_patterns=config['patterns'], device='cuda:0')

        t0 = time.time()
        out = flashvsr(pipe=pipe, frames=frames_tensor, scale=2.0,
                        color_fix=True, color_fix_method="wavelet",
                        tiled_vae=False, tiled_dit=False, tile_size=256,
                        tile_overlap=16, unload_dit=False, sparse_ratio=0.5,
                        kv_ratio=0.5, local_range=128, seed=seed,
                        force_offload=False, enable_debug=False,
                        chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
        elapsed = time.time() - t0
        psnr = (20 * math.log10(1.0 / math.sqrt(
            torch.mean((baseline_out - out).float() ** 2).item()))
            if not torch.isnan(out).any() else float('nan'))
        print(f"Config {config_name} PSNR: {psnr:.2f}dB, Time: {elapsed:.2f}s")
        results[config_name] = {
            'psnr': psnr,
            'time': elapsed,
            'description': config['description'],
            'patterns': config['patterns']
        }
        del pipe, dit
        torch.cuda.empty_cache()

    return results, baseline_time


def main():
    parser = argparse.ArgumentParser(description="PR #2: Mixed-Precision A16 Strategy")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    video_path = args.video or os.path.join(
        project_root, "datasets", "test", "VideoLQ", "LQ-Video", "006.mkv")

    dtype = torch.bfloat16

    print(f"PR #2: Mixed-Precision A16 Strategy")
    print(f"Video: {video_path}")
    print(f"Frames: {args.num_frames}, Max size: {args.max_size}")

    results, baseline_time = test_mixed_precision_configs(
        video_path=video_path,
        num_frames=args.num_frames,
        max_size=args.max_size,
        seed=args.seed,
        dtype=dtype)

    # Summary
    print("\n" + "="*60)
    print("PR #2 RESULTS SUMMARY")
    print("="*60)
    print(f"{'Config':<30} {'PSNR (dB)':<12} {'Time (s)':<10}")
    print("-" * 52)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['psnr']):
        print(f"{name:<30} {r['psnr']:<12.2f} {r['time']:<10.2f}")
    print("-" * 52)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"/home/user/logs/pr2_mixed_precision_{timestamp}.json"
    with open(log_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'video': video_path,
            'num_frames': args.num_frames,
            'max_size': args.max_size,
            'seed': args.seed,
            'baseline_time': baseline_time,
            'results': results
        }, f, indent=2)
    print(f"\nSaved to: {log_path}")


if __name__ == "__main__":
    main()
