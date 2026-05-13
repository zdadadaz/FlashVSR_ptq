#!/usr/bin/env python3
"""
PR #1: Temporal-Aware Activation Calibration for FlashVSR_Integrated W8A8 PTQ

Goal: Replace max-based amax with percentile-based scale estimation,
      compute per-layer temporal CV, and use dynamic alpha per layer type.

Key improvements over v0 (max-based):
1. ObserverLinear: collect per-frame stats (not just running max)
2. Percentile-based scale: use 99th percentile instead of max
3. Temporal-aware CV: measure temporal variation per channel
4. Dynamic alpha: attention QKV use alpha=0.3 (migration-heavy),
                  FFN uses alpha=0.7 (migration-light)

Reference: QuantVSR (STCA), PMQ-VE, PTQ4DiT

Author: localpc
Date: 2026-05-06
"""

import argparse
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock ComfyUI modules
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
from src.models.quantization.smoothquant import (
    ObserverLinear, inject_observers, collect_activation_stats,
    load_video_frames, calculate_smoothquant_scales
)



# ─────────────────────────────────────────────────────────────────
# Enhanced ObserverLinear with temporal-aware stats
# ─────────────────────────────────────────────────────────────────

class TemporalObserverLinear(nn.Module):
    """
    Enhanced observer that collects:
    - Running max (for compatibility)
    - Per-frame amax per channel (for temporal profiling)
    - Per-channel mean/std across all observations
    - 99th/95th/90th percentile per channel
    """

    def __init__(self, linear_module, num_percentiles=3):
        super().__init__()
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.weight = linear_module.weight
        self.bias = linear_module.bias

        # Standard max-based observer (for baseline comparison)
        self.register_buffer("act_amax", torch.zeros(self.in_features, dtype=torch.float32))

        # Temporal profiling: store per-frame amax per channel
        # Shape: [num_frames, in_features] — store last N frame maxima
        self._frame_max_buffer = []
        self._max_frames = 32  # max frames to buffer

        # Per-channel stats accumulators
        self.register_buffer("act_sum", torch.zeros(self.in_features, dtype=torch.float32))
        self.register_buffer("act_sq_sum", torch.zeros(self.in_features, dtype=torch.float32))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))

        # Per-channel percentile accumulators
        self._percentile_buffer = []  # list of [in_features] tensors per batch

    def forward(self, x):
        x_flat = x.view(-1, x.shape[-1])  # [batch*spatial, in_features]
        batch_amax = torch.amax(torch.abs(x_flat), dim=0)  # [in_features]

        # Update running max (original method)
        self.act_amax = torch.max(self.act_amax, batch_amax.float())

        # Per-frame profiling: store frame-level max
        frame_max = torch.amax(torch.abs(x_flat), dim=0)  # [in_features]
        self._frame_max_buffer.append(frame_max.float().cpu())
        if len(self._frame_max_buffer) > self._max_frames:
            self._frame_max_buffer.pop(0)

        # Update per-channel sum for mean/std
        self.act_sum += batch_amax.float()
        self.act_sq_sum += (batch_amax.float() ** 2)
        self.num_batches += 1

        # Store for percentile
        self._percentile_buffer.append(batch_amax.float().cpu())
        if len(self._percentile_buffer) > self._max_frames:
            self._percentile_buffer.pop(0)

        return F.linear(x, self.weight, self.bias)

    def get_temporal_stats(self):
        """Compute temporal CV (coefficient of variation) across frames."""
        if len(self._frame_max_buffer) < 2:
            return {}

        # Stack: [T, in_features]
        frame_maxima = torch.stack(self._frame_max_buffer, dim=0).cuda()
        temporal_mean = torch.mean(frame_maxima, dim=0)  # [in_features]
        temporal_std = torch.std(frame_maxima, dim=0)    # [in_features]

        # CV = std / mean — high CV means large temporal variation
        # Avoid division by zero
        cv = temporal_std / (temporal_mean + 1e-8)

        return {
            'temporal_mean': temporal_mean,
            'temporal_std': temporal_std,
            'cv': cv,  # coefficient of variation per channel
            'cv_global': (temporal_std.sum() / (temporal_mean.sum() + 1e-8)).item(),
            'num_frames': len(self._frame_max_buffer),
        }

    def get_percentile_stats(self, percentiles=(99, 95, 90)):
        """Compute percentile-based scales per channel."""
        if len(self._percentile_buffer) < 1:
            return {}

        all_vals = torch.stack(self._percentile_buffer, dim=0).cuda()  # [T, in_features]
        results = {}
        for p in percentiles:
            pct = torch.quantile(all_vals, p / 100.0, dim=0)  # [in_features]
            results[f'amax_pct{p}'] = pct
            results[f'scale_pct{p}'] = (pct / 124.0).clamp(min=1e-6)
        return results


def calculate_temporal_aware_scales(model, alpha_default=0.5, percentile=99):
    """
    Enhanced scale computation with:
    - Percentile-based scale (instead of max)
    - Per-layer-type alpha (attention vs FFN)
    - Temporal CV scoring
    """
    scales_dict = {}
    meta_dict = {}  # extra info per layer

    for name, module in model.named_modules():
        if isinstance(module, (ObserverLinear, TemporalObserverLinear)):
            act_amax = module.act_amax
            weight_amax = torch.amax(torch.abs(module.weight), dim=0).float()

            # ── Percentile-based activation scale ──
            pct_stats = {}
            if isinstance(module, TemporalObserverLinear):
                pct_stats = module.get_percentile_stats(percentiles=(99, 95, 90))

            if pct_stats:
                # Use 99th percentile as activation scale
                act_scale_pct = pct_stats.get('amax_pct99', act_amax)
            else:
                # Fallback: use max (original method)
                act_scale_pct = act_amax

            # ── Determine alpha based on layer type ──
            # Higher alpha → more migration to weights (act gets harder quantization)
            # Attention QKV: more sensitive → more migration to weights (alpha ↑)
            # FFN: less sensitive → less migration (alpha ↓)
            layer_type = _classify_layer(name)
            if layer_type == 'attention_qkv':
                alpha = 0.3  # more migration to weights, keep activations easier
            elif layer_type == 'attention_o':
                alpha = 0.4
            elif layer_type == 'ffn':
                alpha = 0.7  # less migration, FFN activations are stable
            elif layer_type == 'embedding':
                alpha = 0.5
            else:
                alpha = alpha_default

            # SmoothQuant formula: s = act_max^alpha / weight_max^(1-alpha)
            scale = (torch.pow(act_scale_pct, alpha) / torch.pow(weight_amax, 1.0 - alpha))
            scale = torch.clamp(scale, min=1e-5)

            scales_dict[name] = scale

            # ── Temporal CV ──
            cv_global = 0.0
            temporal_cv = None
            if isinstance(module, TemporalObserverLinear):
                tstats = module.get_temporal_stats()
                cv_global = tstats.get('cv_global', 0.0)
                temporal_cv = tstats.get('cv', None)

            meta_dict[name] = {
                'alpha': alpha,
                'layer_type': layer_type,
                'cv_global': cv_global,
                'act_amax_max': act_amax.max().item(),
                'act_amax_p99': act_scale_pct.max().item() if pct_stats else act_amax.max().item(),
            }

    return scales_dict, meta_dict


def _classify_layer(name):
    """Classify layer type from its name."""
    name_lower = name.lower()
    if 'self_attn' in name_lower or 'selfattn' in name_lower:
        if '.q' in name_lower or '_q' in name_lower or 'query' in name_lower:
            return 'attention_qkv'
        elif '.k' in name_lower or '_k' in name_lower or 'key' in name_lower:
            return 'attention_qkv'
        elif '.v' in name_lower or '_v' in name_lower or 'value' in name_lower:
            return 'attention_qkv'
        elif '.o' in name_lower or '_o' in name_lower or 'output' in name_lower:
            return 'attention_o'
        else:
            return 'attention_other'
    elif 'cross_attn' in name_lower or 'crossattn' in name_lower:
        if '.q' in name_lower or '_q' in name_lower or 'query' in name_lower:
            return 'attention_qkv'
        elif '.k' in name_lower or '_k' in name_lower or 'key' in name_lower:
            return 'attention_qkv'
        elif '.v' in name_lower or '_v' in name_lower or 'value' in name_lower:
            return 'attention_qkv'
        elif '.o' in name_lower or '_o' in name_lower or 'output' in name_lower:
            return 'attention_o'
        else:
            return 'attention_other'
    elif 'ffn' in name_lower or 'mlp' in name_lower:
        return 'ffn'
    elif 'text_embedding' in name_lower or 'time_embedding' in name_lower or 'head' in name_lower:
        return 'embedding'
    else:
        return 'other'


# ─────────────────────────────────────────────────────────────────
# Calibration with TemporalObserverLinear
# ─────────────────────────────────────────────────────────────────

def inject_temporal_observers(model):
    """"Replace nn.Linear with TemporalObserverLinear for enhanced calibration.
    
    Uses iteration instead of recursion to avoid deep stack.
    """
    # Collect all (parent, name, module) tuples for modules that need wrapping
    # before modifying anything
    to_wrap = []
    for parent_name, parent_mod in model.named_modules():
        prefix = parent_name + "." if parent_name else ""
        for child_name, child_mod in parent_mod.named_children():
            full_name = prefix + child_name
            if hasattr(child_mod, 'act_amax'):
                # Already wrapped, skip
                continue
            if isinstance(child_mod, nn.Linear):
                to_wrap.append((parent_mod, child_name, child_mod, full_name))
    
    # Now apply replacements
    for parent_mod, child_name, child_mod, full_name in to_wrap:
        wrapped = TemporalObserverLinear(child_mod)
        setattr(parent_mod, child_name, wrapped)
    
    return model


def collect_temporal_activation_stats(model, dataset_path, pipe, num_videos=3, frames_per_video=4, max_size=128):
    """
    Run calibration with TemporalObserverLinear modules to collect
    temporal-aware activation statistics.
    """
    from torch.nn import functional as F

    model = inject_temporal_observers(model)
    model.cuda()
    model.eval()

    # Initialize cross KV cache
    try:
        if hasattr(pipe, 'init_cross_kv'):
            dummy_ctx = torch.randn(1, 10, 4096, device='cuda')
            pipe.init_cross_kv(context_tensor=dummy_ctx)
    except:
        pass

    # Find calibration videos
    video_dirs = [
        ("SPMCS", "LQ-Video"),
        ("VideoLQ", "LQ-Video"),
        ("RealVSR", "LQ-Video"),
        ("UDM10", "LQ-Video"),
        ("YouHQ40", "LQ-Video"),
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
        return {}, {}

    print(f"PR#1 Temporal Calibration: {len(videos)} videos × {frames_per_video} frames")

    for video_path in videos[:num_videos]:
        print(f"  Processing: {os.path.basename(video_path)}")
        frames = load_video_frames(video_path, num_frames=frames_per_video, max_size=max_size)
        if len(frames) < 2:
            continue

        frames_np = np.stack(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)

        try:
            with torch.no_grad():
                _ = flashvsr(
                    pipe=pipe, frames=frames_tensor,
                    scale=2.0, color_fix=True, color_fix_method="wavelet",
                    tiled_vae=False, tiled_dit=False, tile_size=256, tile_overlap=16,
                    unload_dit=False, sparse_ratio=0.5, kv_ratio=0.5, local_range=128,
                    seed=42, force_offload=False, enable_debug=False,
                    chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0
                )
        except Exception as e:
            print(f"    Warning: inference failed: {e}")
            continue

    # Collect stats from all TemporalObserverLinear modules
    act_stats = {}
    meta_info = {}
    for name, module in model.named_modules():
        if isinstance(module, TemporalObserverLinear):
            act_stats[name] = module.act_amax.clone()
            module.eval()
            # Compute temporal stats for this layer
            tstats = module.get_temporal_stats()
            pct_stats = module.get_percentile_stats(percentiles=(99, 95, 90))
            meta_info[name] = {
                'temporal_cv_global': tstats.get('cv_global', 0.0),
                'num_frames': tstats.get('num_frames', 0),
                'act_amax_max': module.act_amax.max().item(),
                'act_amax_mean': module.act_amax.mean().item(),
                'layer_type': _classify_layer(name),
            }
            if pct_stats:
                meta_info[name]['act_amax_p99'] = pct_stats['amax_pct99'].max().item()
                meta_info[name]['act_amax_p95'] = pct_stats['amax_pct95'].max().item()

    # Restore original nn.Linear modules
    for name, module in model.named_children():
        if isinstance(module, TemporalObserverLinear):
            new_linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_linear.weight = nn.Parameter(module.weight.data.clone())
            if module.bias is not None:
                new_linear.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new_linear)

    return act_stats, meta_info


# ─────────────────────────────────────────────────────────────────
# Compare old vs new calibration methods
# ─────────────────────────────────────────────────────────────────

def test_calibration_methods(video_path, num_frames=4, max_size=128, seed=42):
    """
    Compare three calibration methods:
    A. Original: max-based, alpha=0.5 fixed
    B. Percentile only: 99th percentile, alpha=0.5 fixed
    C. Temporal-aware: percentile + dynamic alpha per layer type
    """
    from torch.nn import functional as F

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # Load video
    frames = load_video_frames(video_path, num_frames=num_frames, max_size=max_size)
    frames_np = np.stack(frames).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_np)

    # ── FP16 Baseline ──
    print("\n" + "="*60)
    print("FP16 Baseline")
    print("="*60)
    pipe_fp16 = init_pipeline(model="FlashVSR", mode="tiny", device="cuda:0",
                            dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    t0 = time.time()
    baseline_out = flashvsr(pipe=pipe_fp16, frames=frames_tensor, scale=2.0,
                             color_fix=True, color_fix_method="wavelet",
                             tiled_vae=False, tiled_dit=False, tile_size=256, tile_overlap=16,
                             unload_dit=False, sparse_ratio=0.5, kv_ratio=0.5, local_range=128,
                             seed=seed, force_offload=False, enable_debug=False,
                             chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
    baseline_time = time.time() - t0
    print(f"FP16: {baseline_time:.2f}s")
    del pipe_fp16
    torch.cuda.empty_cache()

    # ── Method A: Original SmoothQuant (max, alpha=0.5) ──
    print("\n" + "="*60)
    print("Method A: Original SmoothQuant (max, alpha=0.5)")
    print("="*60)
    pipe_A = init_pipeline(model="FlashVSR", mode="tiny", device="cuda:0",
                           dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    dit_A = pipe_A.denoising_model()
    dit_A = inject_observers(dit_A)
    if hasattr(pipe_A, 'init_cross_kv'):
        pipe_A.init_cross_kv(context_tensor=torch.randn(1, 10, 4096, device='cuda:0'))

    # Run calibration forward pass
    frames_A = load_video_frames(video_path, num_frames=num_frames, max_size=max_size)
    frames_np_A = np.stack(frames_A).astype(np.float32) / 255.0
    frames_tensor_A = torch.from_numpy(frames_np_A)
    try:
        with torch.no_grad():
            _ = flashvsr(pipe=pipe_A, frames=frames_tensor_A, scale=2.0,
                          color_fix=True, color_fix_method="wavelet",
                          tiled_vae=False, tiled_dit=False, tile_size=256, tile_overlap=16,
                          unload_dit=False, sparse_ratio=0.5, kv_ratio=0.5, local_range=128,
                          seed=seed, force_offload=False, enable_debug=False,
                          chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
    except: pass

    # Convert using original method — collect act_amax from dit_A's ObserverLinear buffers
    from src.models.quantization.quant import convert_model_to_w8a8_smoothquant
    dit_A_conv = pipe_A.denoising_model()  # fresh copy (no observers)
    dit_A_conv = dit_A_conv.cuda()  # ensure it's on CUDA before conversion
    # Extract act_amax from dit_A's ObserverLinear modules
    act_stats_A = {}
    for name, module in dit_A.named_modules():
        if hasattr(module, 'act_amax'):
            act_stats_A[name] = module.act_amax.clone().cuda()
    dit_A_conv = convert_model_to_w8a8_smoothquant(dit_A_conv, act_stats=act_stats_A, alpha=0.5, method='max')

    t0 = time.time()
    out_A = flashvsr(pipe=pipe_A, frames=frames_tensor, scale=2.0,
                      color_fix=True, color_fix_method="wavelet",
                      tiled_vae=False, tiled_dit=False, tile_size=256, tile_overlap=16,
                      unload_dit=False, sparse_ratio=0.5, kv_ratio=0.5, local_range=128,
                      seed=seed, force_offload=False, enable_debug=False,
                      chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
    time_A = time.time() - t0
    psnr_A = 20 * math.log10(1.0 / math.sqrt(torch.mean((baseline_out - out_A).float() ** 2).item())) if not torch.isnan(out_A).any() else float('nan')
    print(f"Method A PSNR: {psnr_A:.2f}dB, Time: {time_A:.2f}s")
    del pipe_A, dit_A, dit_A_conv
    torch.cuda.empty_cache()

    # ── Method C: Temporal-Aware Calibration ──
    print("\n" + "="*60)
    print("Method C: Temporal-Aware (percentile + dynamic alpha)")
    print("="*60)
    pipe_C = init_pipeline(model="FlashVSR", mode="tiny", device="cuda:0",
                           dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    dit_C = pipe_C.denoising_model()

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "test")
    act_stats_C, meta_C = collect_temporal_activation_stats(
        dit_C, dataset_path, pipe_C, num_videos=3, frames_per_video=4, max_size=max_size)

    print(f"\nCollected stats from {len(act_stats_C)} layers")
    scales_C, meta_scales = calculate_temporal_aware_scales(dit_C, alpha_default=0.5, percentile=99)

    # Show per-layer alpha distribution
    alpha_dist = {}
    for name, m in meta_scales.items():
        lt = m['layer_type']
        alpha_dist[lt] = alpha_dist.get(lt, []) + [m['alpha']]

    print("\nPer-layer-type alpha distribution:")
    for lt, alphas in sorted(alpha_dist.items()):
        print(f"  {lt}: alpha={alphas[0]:.1f} ({len(alphas)} layers)")

    # Show CV stats
    cv_stats = {}
    for name, m in meta_C.items():
        lt = m['layer_type']
        if lt not in cv_stats:
            cv_stats[lt] = []
        cv_stats[lt].append(m['temporal_cv_global'])

    print("\nTemporal CV (coefficient of variation) per layer type:")
    for lt, cvs in sorted(cv_stats.items()):
        print(f"  {lt}: CV={np.mean(cvs):.4f}±{np.std(cvs):.4f} ({len(cvs)} layers)")

    # Convert with temporal-aware method
    dit_C_conv = pipe_C.denoising_model()  # fresh copy
    dit_C_conv = convert_model_to_w8a8_smoothquant(dit_C_conv, scales_C, alpha=0.5, method='max')

    t0 = time.time()
    out_C = flashvsr(pipe=pipe_C, frames=frames_tensor, scale=2.0,
                      color_fix=True, color_fix_method="wavelet",
                      tiled_vae=False, tiled_dit=False, tile_size=256, tile_overlap=16,
                      unload_dit=False, sparse_ratio=0.5, kv_ratio=0.5, local_range=128,
                      seed=seed, force_offload=False, enable_debug=False,
                      chunk_size=0, resize_factor=1.0, mode="tiny", context_pad=0)
    time_C = time.time() - t0
    psnr_C = 20 * math.log10(1.0 / math.sqrt(torch.mean((baseline_out - out_C).float() ** 2).item())) if not torch.isnan(out_C).any() else float('nan')
    print(f"Method C PSNR: {psnr_C:.2f}dB, Time: {time_C:.2f}s")

    del pipe_C, dit_C, dit_C_conv
    torch.cuda.empty_cache()

    return {
        'fp16': {'psnr': float('inf'), 'time': baseline_time},
        'method_A_max_alpha05': {'psnr': psnr_A, 'time': time_A},
        'method_C_temporal': {'psnr': psnr_C, 'time': time_C},
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PR #1: Temporal-Aware Calibration")
    parser.add_argument("--model", type=str, default="FlashVSR")
    parser.add_argument("--mode", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str,
        default="/home/user/apps/FlashVSRptq/FlashVSR_Integrated/datasets/test")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=128)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Find video
    if args.video and os.path.exists(args.video):
        video_path = args.video
    else:
        for dataset in ["VideoLQ", "SPMCS"]:
            lq_path = Path(args.dataset) / dataset / "LQ-Video"
            if lq_path.exists():
                videos = list(lq_path.glob("*.mkv")) + list(lq_path.glob("*.mp4"))
                if videos:
                    video_path = str(videos[0])
                    break
        else:
            raise FileNotFoundError(f"No videos found in {args.dataset}")

    print(f"PR #1: Temporal-Aware Calibration")
    print(f"Video: {video_path}")
    print(f"Frames: {args.num_frames}, Max size: {args.max_size}")
    print(f"Calibration videos: {args.num_videos}")

    results = test_calibration_methods(video_path, args.num_frames, args.max_size, args.seed)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method, res in results.items():
        print(f"  {method}: PSNR={res['psnr']:.2f}dB, Time={res['time']:.2f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/home/user/logs/pr1_temporal_calibration_{ts}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'video': video_path,
            'num_frames': args.num_frames,
            'num_calibration_videos': args.num_videos,
            'results': {k: {'psnr': v['psnr'], 'time': v['time']} for k, v in results.items()}
        }, f, indent=2, default=lambda x: float('nan') if isinstance(x, float) and math.isnan(x) else x)
    print(f"\nSaved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
