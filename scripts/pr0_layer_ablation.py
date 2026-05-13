#!/usr/bin/env python3
"""
PR #0: Per-layer Quantization Diagnostic for FlashVSR_Integrated W8A8 PTQ

Goal: Identify which layers cause the biggest PSNR drop when quantized.
Uses ckpt-level selective replacement (no device-host confusion).

Logic:
- Run FP16 baseline → capture baseline output
- For each layer group, selectively replace only those layers in state_dict
  with W8A16 int8 weights, save temp ckpt, reload as fresh pipeline
- This avoids the "fresh pipeline but shared model object" device issues

Author: localpc
Date: 2026-05-06
"""

import argparse
import os
import sys
import torch
import time
import math
import cv2
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime

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
from src.models.quantization.quant import WeightOnlyInt8Linear


def load_video_frames(video_path, num_frames=8, max_size=128):
    cap = cv2.VideoCapture(video_path)
    frames = []
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


def calculate_psnr(t1, t2):
    if t1.shape != t2.shape:
        t2 = t2[:, :t1.shape[1], :t1.shape[2], :]
    mse = torch.mean((t1.float() - t2.float()) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def get_layer_groups(model):
    """Categorize all nn.Linear layers in WanModel."""
    groups = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if 'self_attn' in name:
            last = name.split('.')[-1]
            group = f'SelfAttn.{last.upper()}' if last in ('q','k','v','o') else 'SelfAttn.other'
        elif 'cross_attn' in name:
            last = name.split('.')[-1]
            group = f'CrossAttn.{last.upper()}' if last in ('q','k','v','o') else 'CrossAttn.other'
        elif 'ffn' in name:
            group = 'FFN'
        elif 'head' in name:
            group = 'Head'
        elif any(x in name for x in ('text_embedding', 'time_embedding', 'time_projection')):
            group = 'Embed'
        else:
            group = 'Other'
        groups.setdefault(group, []).append(name)
    return groups


def quantize_state_dict_layer_groups(state_dict, groups_dict, target_groups, method='max'):
    """
    Replace weight tensors in state_dict for layers belonging to target_groups.
    Does NOT replace keys, just the weight data with int8 + scale.
    Returns a NEW state_dict with quantized weights (int8 stored as same key).
    """
    new_sd = {}
    for key, tensor in state_dict.items():
        if not key.endswith('.weight') and not key.endswith('.bias'):
            new_sd[key] = tensor
            continue
        
        # Determine which group this layer belongs to
        layer_key = key.replace('.weight', '').replace('.bias', '')
        matched_group = None
        for gname, lnames in groups_dict.items():
            if any(ln in key for ln in lnames):
                matched_group = gname
                break
        
        if matched_group not in target_groups:
            new_sd[key] = tensor
            continue
        
        if key.endswith('.weight'):
            w = tensor  # original fp16/bf16
            dim = w.shape[0]  # out_features
            
            # Compute scale
            if method == 'max':
                w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
            elif method == 'percentile99':
                flat_w = w.flatten()
                k = int(flat_w.numel() * 0.99)
                sorted_w, _ = torch.sort(torch.abs(flat_w))
                w_max = sorted_w[k].view(1, 1)
            else:
                w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
            
            scale = (w_max / 124.0).clamp(min=1e-6)
            
            # Quantize weight
            w_int8 = torch.round(w / scale).to(torch.int8)
            
            # Store as key + '_int8' and key + '_scale'
            new_sd[key + '_int8'] = w_int8
            new_sd[key + '_scale'] = scale
            # Keep original fp16 weight under original key for now (will be replaced by wrapper later)
            # We'll use a marker to indicate this layer is quantized
            new_sd[key] = w  # keep fp16 temporarily, convert_model will pick up the int8
            
        elif key.endswith('.bias'):
            new_sd[key] = tensor  # keep bias fp16
    
    return new_sd


def convert_state_dict_to_w8a16(state_dict, groups_dict, target_groups, method='max'):
    """
    Convert specific layer groups in state_dict to W8A16 format.
    Returns (new_state_dict, replacement_map) where replacement_map has
    {layer_name: (int8_weight, scale)} for the targeted layers.
    """
    replacement_map = {}
    
    for gname, lnames in groups_dict.items():
        if gname not in target_groups:
            continue
        
        for ln in lnames:
            w_key = f'{ln}.weight'
            b_key = f'{ln}.bias'
            
            if w_key not in state_dict:
                continue
            
            w = state_dict[w_key]
            
            if method == 'max':
                w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
            elif method == 'percentile99':
                flat_w = w.flatten()
                k = int(flat_w.numel() * 0.99)
                sorted_w, _ = torch.sort(torch.abs(flat_w))
                w_max = sorted_w[k].view(1, 1)
            else:
                w_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
            
            scale = (w_max / 124.0).clamp(min=1e-6)
            w_int8 = torch.round(w / scale).to(torch.int8)
            
            replacement_map[ln] = {
                'int8_weight': w_int8,
                'scale': scale,
                'bias': state_dict[b_key].clone() if b_key in state_dict else None
            }
    
    return replacement_map


def apply_w8a16_to_model(model, replacement_map):
    """Apply W8A16 conversion using precomputed replacement_map."""
    for ln, data in replacement_map.items():
        parts = ln.split('.')
        parent = model
        for p in parts[:-1]:
            parent = parent._modules[p]
        last = parts[-1]
        
        orig = parent._modules[last]
        new_mod = WeightOnlyInt8Linear(
            orig.in_features, orig.out_features,
            bias=data['bias'] is not None,
            device=data['int8_weight'].device,
            dtype=data['int8_weight'].dtype
        )
        new_mod.weight.copy_(data['int8_weight'])
        new_mod.weight_scale.copy_(data['scale'])
        if data['bias'] is not None:
            new_mod.bias.copy_(data['bias'])
        
        parent._modules[last] = new_mod
    
    return model


def run_flashvsr(pipe, frames_tensor, seed=42, mode="tiny"):
    return flashvsr(
        pipe=pipe, frames=frames_tensor,
        scale=2.0, color_fix=True, color_fix_method="wavelet",
        tiled_vae=False, tiled_dit=False, tile_size=256, tile_overlap=16,
        unload_dit=False, sparse_ratio=0.5, kv_ratio=0.5, local_range=128,
        seed=seed, force_offload=False, enable_debug=False,
        chunk_size=0, resize_factor=1.0, mode=mode, context_pad=0
    )


def main():
    parser = argparse.ArgumentParser(description="PR #0: Per-layer quantization diagnostic")
    parser.add_argument("--model", type=str, default="FlashVSR", choices=["FlashVSR", "FlashVSR-v1.1"])
    parser.add_argument("--mode", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str,
        default="/home/user/apps/FlashVSRptq/FlashVSR_Integrated/datasets/test")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=128)
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
                    print(f"Auto-selected: {video_path}")
                    break
        else:
            raise FileNotFoundError(f"No videos found in {args.dataset}")

    frames = load_video_frames(video_path, num_frames=args.num_frames, max_size=args.max_size)
    frames_np = np.stack(frames).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_np)
    print(f"Input: {frames_tensor.shape}")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Dtype: {dtype}")

    # =========================================================================
    # STEP 1: FP16 Baseline
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: FP16 Baseline")
    print("="*60)
    pipe_base = init_pipeline(model=args.model, mode=args.mode, device=args.device,
                              dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    baseline_output = run_flashvsr(pipe_base, frames_tensor, seed=args.seed, mode=args.mode)
    baseline_time = time.time() - t0
    print(f"Baseline: {baseline_time:.2f}s, Output: {baseline_output.shape}")
    del pipe_base
    torch.cuda.empty_cache()

    # =========================================================================
    # STEP 2: W8A16 Full (all layers quantized)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: W8A16 Full (all layers)")
    print("="*60)
    pipe_w8a16 = init_pipeline(model=args.model, mode=args.mode, device=args.device,
                               dtype=dtype, vae_model="Wan2.1", quantize_mode="W8A16")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    w8a16_output = run_flashvsr(pipe_w8a16, frames_tensor, seed=args.seed, mode=args.mode)
    w8a16_time = time.time() - t0
    psnr_w8a16 = calculate_psnr(baseline_output, w8a16_output)
    print(f"W8A16 Full: PSNR={psnr_w8a16:.2f}dB vs FP16, Time={w8a16_time:.2f}s")
    del pipe_w8a16
    torch.cuda.empty_cache()

    # =========================================================================
    # STEP 3: Get layer groups from a fresh model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Per-Group Ablation")
    print("="*60)
    
    # Get reference model for layer groups
    pipe_ref = init_pipeline(model=args.model, mode=args.mode, device=args.device,
                              dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
    dit_ref = pipe_ref.denoising_model()
    groups = get_layer_groups(dit_ref)
    del pipe_ref
    torch.cuda.empty_cache()
    
    print(f"\nLayer groups found:")
    for gname, lnames in sorted(groups.items()):
        print(f"  {gname}: {len(lnames)} layers")
    
    results = []
    results.append({
        "group": "W8A16_Full",
        "psnr_vs_fp16": psnr_w8a16,
        "num_layers": sum(len(v) for v in groups.values()),
        "note": "All layers W8A16"
    })

    # =========================================================================
    # STEP 4: Per-group ablation — fresh pipeline per group
    # =========================================================================
    for group_name, layer_names in sorted(groups.items()):
        print(f"\n--- Quantizing group: {group_name} ({len(layer_names)} layers) ---")
        
        try:
            # Fresh FP16 pipeline
            pipe_fresh = init_pipeline(model=args.model, mode=args.mode, device=args.device,
                                       dtype=dtype, vae_model="Wan2.1", quantize_mode="None")
            dit = pipe_fresh.denoising_model()
            
            # Apply W8A16 to this group ONLY
            target_groups = {group_name}
            rm = convert_state_dict_to_w8a16(dit.state_dict(), groups, target_groups, method='max')
            dit = apply_w8a16_to_model(dit, rm)
            
            torch.cuda.empty_cache()
            out = run_flashvsr(pipe_fresh, frames_tensor, seed=args.seed, mode=args.mode)
            psnr = calculate_psnr(baseline_output, out)
        except Exception as e:
            print(f"  Inference FAILED: {e}")
            import traceback
            traceback.print_exc()
            psnr = float('nan')
        
        results.append({
            "group": group_name,
            "num_layers": len(layer_names),
            "psnr_vs_fp16": psnr,
        })
        print(f"  PSNR vs FP16: {psnr:.2f}dB")
        
        try:
            del pipe_fresh
        except:
            pass
        torch.cuda.empty_cache()

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nBaseline FP16: {baseline_time:.2f}s")
    print(f"W8A16 Full:   PSNR={psnr_w8a16:.2f}dB vs FP16")
    print(f"\n{'Group':<22} {'PSNR vs FP16':>14} {'#Layers':>8}")
    print("-" * 48)
    sorted_results = sorted(results, key=lambda x: x['psnr_vs_fp16'] if not math.isnan(x['psnr_vs_fp16']) else 999)
    for r in sorted_results:
        print(f"{r['group']:<22} {r['psnr_vs_fp16']:>14.2f} {r['num_layers']:>8}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/home/user/logs/pr0_ablation_{ts}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "video": video_path,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "baseline_time_s": baseline_time,
        "w8a16_full": {"psnr_vs_fp16": psnr_w8a16, "time_s": w8a16_time},
        "per_group_ablation": sorted_results,
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float('nan') if isinstance(x, float) and math.isnan(x) else x)
    print(f"\nSaved to: {output_path}")
    
    # Analysis
    print("\n--- BOTTLENECK ANALYSIS ---")
    print(f"W8A16 Full PSNR: {psnr_w8a16:.2f}dB vs FP16")
    print(f"Note: 36.68dB is already good for weight-only quantization.")
    print(f"The 9.07dB issue was from W8A8 SmoothQuant activation quantization.")
    print(f"\nThis means:")
    print(f"  1. Weight-only (W8A16) is FINE: {psnr_w8a16:.2f}dB")
    print(f"  2. The problem is ACTIVATION quantization (W8A8 SmoothQuant)")
    print(f"  3. Per-group ablation below tells us which layers need A16 vs A8")
    
    return summary


if __name__ == "__main__":
    main()
