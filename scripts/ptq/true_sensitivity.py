"""End-to-end per-layer A8W8 sensitivity measurement for FlashVSR DiT.

對每個 Linear layer:
  1. 用 FP16 input 跑 forward, 存 FP16 reference output
  2. 把該 layer 換成 FakeQuantLinear (dynamic_asymmetric A8W8)
  3. 重新跑 forward, 量 MSE 跟 relative L1
  4. 還原成 nn.Linear

比 6/10 sensitivity report (mu_var 啟發式) 準確, 因為這是真實 quantization error。
不會改動 model 的 state_dict, run 完 hook 直接卸載。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make repo root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.quantization.fakequant import FakeQuantLinear
from scripts.ptq.fakequant_calibrate import (
    build_dit,
    load_checkpoint,
    discover_calibration_videos,
    sample_latents_from_video,
)


# -----------------------------------------------------------------------------
# 1. 跑一次 forward, 用 hook 攔截每個 Linear 的 (input, output) tensor
# -----------------------------------------------------------------------------

class _CaptureBuffers:
    """存每個 Linear layer 的 FP16 input + output."""
    def __init__(self):
        self.inputs: Dict[str, torch.Tensor] = {}
        self.outputs: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: nn.Module):
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            h_in = mod.register_forward_pre_hook(self._make_input_hook(name))
            h_out = mod.register_forward_hook(self._make_output_hook(name))
            self._handles.extend([h_in, h_out])

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_input_hook(self, name: str):
        def hook(module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            # 存 detach 的 fp16 tensor, 避免 autograd 累積
            self.inputs[name] = x.detach().to(torch.float32).cpu()
        return hook

    def _make_output_hook(self, name: str):
        def hook(module, inputs, output):
            y = output if isinstance(output, torch.Tensor) else output[0]
            self.outputs[name] = y.detach().to(torch.float32).cpu()
        return hook


# -----------------------------------------------------------------------------
# 2. 對單一層量 FP vs FakeQuant 的 MSE
# -----------------------------------------------------------------------------

@torch.no_grad()
def _quantize_layer_error(
    layer: nn.Linear,
    x_fp32: torch.Tensor,
    y_fp32: torch.Tensor,
    quant_mode: str = "a8w8_dynamic_asymmetric",
) -> Tuple[float, float]:
    """把 layer 換成 FakeQuantLinear 算一次, 量 mse 跟 relative L1.

    x_fp32: layer 的 FP input, shape [..., in_features]
    y_fp32: layer 的 FP output, shape [..., out_features]
    return: (mse, rel_l1)
    """
    # Save original
    orig_weight = layer.weight.data.clone()
    orig_bias = layer.bias.data.clone() if layer.bias is not None else None

    try:
        # Replace with FakeQuantLinear
        fq = FakeQuantLinear.from_float(
            layer,
            activation_mode="a8",
            weight_mode="w8",
            activation_qdq_mode="dynamic_asymmetric",
            weight_rounding="nearest",
        )

        # Move to same device as input
        device = x_fp32.device
        fq = fq.to(device)

        # Re-quantize input in fp32 (FakeQuantLinear expects same dtype as x)
        # Hook 攔截的 x 是原始的 dtype, 我們用 fp32 跑 FakeQuant 拿乾淨結果
        x_for_quant = x_fp32.to(device)
        y_fq = fq(x_for_quant)
        y_fq_fp32 = y_fq.to(torch.float32).cpu()

        # MSE
        mse = float(((y_fp32 - y_fq_fp32) ** 2).mean().item())

        # Relative L1: |y_q - y_fp| / (|y_fp| + eps)
        rel_l1 = float(((y_fq_fp32 - y_fp32).abs() / (y_fp32.abs() + 1e-6)).mean().item())

        return mse, rel_l1
    finally:
        # Restore original Linear
        layer.weight.data.copy_(orig_weight)
        if orig_bias is not None and layer.bias is not None:
            layer.bias.data.copy_(orig_bias)


# -----------------------------------------------------------------------------
# 3. 對所有 Linear 跑 sensitivity 量測
# -----------------------------------------------------------------------------

@torch.no_grad()
def measure_sensitivity(
    model: nn.Module,
    latents: torch.Tensor,
    contexts: List[torch.Tensor],
    device: str = "cuda",
    num_samples: int = 16,
) -> Dict[str, Dict[str, float]]:
    """跑 num_samples 次 forward, 對每個 Linear layer 量 mse.

    Args:
        model: WanModel in eval mode (in bf16, on cuda)
        latents: (N, F, C, H, W) tensor of pre-encoded latents (4D after squeeze)
        contexts: list of (1, 10, 4096) text context tensors (raw, pre-text-embedding)
        device: device
        num_samples: 用前 num_samples 個 latents 跑 forward

    Returns:
        {layer_name: {"mse": float, "rel_l1": float}}
    """
    # Step 1: Run forward with hooks to capture FP (input, output) for each layer
    cap = _CaptureBuffers()
    cap.attach(model)

    # Pre-compute t_mod (per fakequant_calibrate.py:1162-1164 pattern)
    from src.models.wan_video_dit import sinusoidal_embedding_1d
    # Cast t_big to model dtype to avoid Linear dtype mismatch
    model_dtype = next(model.parameters()).dtype
    t_big_int = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
    t_emb = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t_big_int.float()).to(model_dtype))
    t_mod_for_fwd = model.time_projection(t_emb).unflatten(1, (6, model.dim))

    # Manual block-level forward (per fakequant_calibrate.py:1169-1216)
    pf, ph, pw = model.patch_size
    win_f, win_h, win_w = 2, 8, 8
    req_f = pf * win_f
    req_h = ph * win_h
    req_w = pw * win_w

    for i in range(min(num_samples, len(latents))):
        # latents is (F, C, H, W) or (B, C, F, H, W) — handle both
        if latents.dim() == 4:
            # (F, C, H, W) — single batch
            x_4d = latents[i:i+1].float()  # (1, F, C, H, W)
        else:
            x_4d = latents[i:i+1].float()
        # To 5D
        if x_4d.dim() == 5:
            x_5d = x_4d
        else:
            x_5d = x_4d.unsqueeze(2)  # (1, C, 1, H, W)
        x_5d = x_5d.to(device=device, dtype=model_dtype)
        _, _, D_pad, H_pad, W_pad = x_5d.shape
        pad_f = (req_f - D_pad % req_f) % req_f
        pad_h = (req_h - H_pad % req_h) % req_h
        pad_w = (req_w - W_pad % req_w) % req_w
        if pad_f or pad_h or pad_w:
            x_5d = torch.nn.functional.pad(x_5d, (0, pad_w, 0, pad_h, 0, pad_f))

        # text context
        ctx = contexts[i % len(contexts)]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        if not ctx.is_cuda:
            ctx = ctx.cuda()
        # Cast to model dtype (text_embedding expects model dtype Linear weights)
        ctx = model.text_embedding(ctx.to(model_dtype))

        # patchify + freqs
        x_patched, (f_, h_, w_) = model.patchify(x_5d)
        freqs_i = torch.cat([
            model.freqs[0][:f_].view(f_, 1, 1, -1).expand(f_, h_, w_, -1),
            model.freqs[1][:h_].view(1, h_, 1, -1).expand(f_, h_, w_, -1),
            model.freqs[2][:w_].view(1, 1, w_, -1).expand(f_, h_, w_, -1),
        ], dim=-1).reshape(f_ * h_ * w_, 1, -1).cuda().float()
        x = x_patched
        try:
            for block in model.blocks:
                x = block(x, ctx, t_mod_for_fwd, freqs_i,
                          f_, h_, w_, f_ * h_ * w_, f_ * h_ * w_,
                          False, i, 1, False, False, None, None)
            # Also run head to capture head.head quant error
            _ = model.head(x, t_emb)
        except Exception as e:
            print(f"  [Sensitivity] forward pass {i} error: {e}")
            continue

    cap.detach()

    print(f"  [Sensitivity] Captured (input, output) for {len(cap.outputs)} Linear layers")

    # Step 2: For each layer, compute FakeQuant vs FP error
    results: Dict[str, Dict[str, float]] = {}
    layer_count = 0
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if name not in cap.inputs or name not in cap.outputs:
            continue
        layer_count += 1
        mse, rel_l1 = _quantize_layer_error(
            layer, cap.inputs[name], cap.outputs[name],
        )
        results[name] = {"mse": mse, "rel_l1": rel_l1}

    print(f"  [Sensitivity] Measured {len(results)} / {layer_count} layers")
    return results


# -----------------------------------------------------------------------------
# 4. 把結果寫到 JSON
# -----------------------------------------------------------------------------

def save_sensitivity_cache(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    metadata: dict,
):
    cache = {
        "_metadata": {
            "schema_version": "flashvsr.true_sensitivity.v1",
            "stats": ["mse", "rel_l1"],
            **metadata,
        },
    }
    # Sort by mse desc, also save rank
    sorted_layers = sorted(results.items(), key=lambda x: x[1]["mse"], reverse=True)
    for name, info in sorted_layers:
        cache[name] = {
            "mse": float(info["mse"]),
            "rel_l1": float(info["rel_l1"]),
        }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[Sensitivity] Saved cache to {output_path}")
    # Save rank separately for easy reading
    print(f"[Sensitivity] Top-30 most sensitive layers:")
    for rank, (name, info) in enumerate(sorted_layers[:30], 1):
        print(f"  {rank:2d}. {name}: mse={info['mse']:.6e}, rel_l1={info['rel_l1']:.4e}")


def load_sensitivity_cache(path: str) -> Dict[str, Dict[str, float]]:
    """Load sensitivity cache, return {layer_name: {mse, rel_l1}} (skip _metadata)."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        out[k] = {"mse": float(v["mse"]), "rel_l1": float(v["rel_l1"])}
    return out


# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="True end-to-end per-layer sensitivity for FlashVSR DiT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FP DiT .safetensors")
    parser.add_argument("--output_cache", type=str, required=True, help="Output JSON path")
    parser.add_argument("--dataset_train", type=str, default="datasets/train")
    parser.add_argument("--num_videos", type=int, default=10, help="Number of calibration videos to sample")
    parser.add_argument("--num_samples", type=int, default=16, help="Forward passes per layer")
    parser.add_argument("--calib_frames", type=int, default=8, help="Frames per video")
    parser.add_argument("--latent_size", type=str, default="60x80", help="HxW")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--vae_model", type=str, default="Wan2.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Parse latent size
    h_str, w_str = args.latent_size.lower().split("x")
    frame_size = (int(h_str), int(w_str))

    # 1. Build model
    print(f"[Sensitivity] Building WanModel...")
    model = build_dit("FlashVSR-v1.1")
    model = load_checkpoint(args.checkpoint, model)
    # Cast entire model to bf16 to match patch_embedding bias dtype (avoids dtype mismatch)
    model = model.to(args.device).to(torch.bfloat16).eval()

    # 2. Sample latents
    selected_videos = discover_calibration_videos(args.dataset_train, args.num_videos, args.seed)
    print(f"[Sensitivity] Selected {len(selected_videos)} videos")
    latent_chunks = []
    for vp in selected_videos:
        print(f"  Sampling {vp}...")
        lat, _ = sample_latents_from_video(
            video_path=vp,
            num_frames=args.calib_frames,
            latent_channels=16,
            frame_size=frame_size,
            vae_path=args.vae_path,
            vae_model=args.vae_model,
        )
        latent_chunks.append(lat)
    latents = torch.cat(latent_chunks, dim=0)
    print(f"[Sensitivity] Total latent samples: {latents.shape}")

    # 3. Build dummy contexts (match model dtype)
    contexts = [
        torch.randn(1, 10, 4096, dtype=torch.bfloat16)
        for _ in range(min(args.num_samples, 10))
    ]

    # 4. Measure sensitivity
    print(f"[Sensitivity] Running {args.num_samples} forward passes...")
    results = measure_sensitivity(
        model, latents, contexts,
        device=args.device, num_samples=args.num_samples,
    )

    # 5. Save
    save_sensitivity_cache(
        results, args.output_cache,
        metadata={
            "checkpoint": args.checkpoint,
            "num_videos": len(selected_videos),
            "num_samples": args.num_samples,
            "calib_frames": args.calib_frames,
            "latent_size": args.latent_size,
            "vae_path": args.vae_path,
            "vae_model": args.vae_model,
            "seed": args.seed,
            "quant_mode": "a8w8_dynamic_asymmetric",
        },
    )


if __name__ == "__main__":
    main()
