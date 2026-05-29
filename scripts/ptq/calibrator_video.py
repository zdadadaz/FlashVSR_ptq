"""
W8A8 PTQ Calibration for FlashVSR-v1.1 using video-based latent sampling.
Uses the video-file method (like fakequant_calibrate.py) to avoid the DOVE dataset hang.
All model and input tensors use bfloat16 for RTX Ampere+ GPU compatibility.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.wan_video_dit import WanModel


class ActivationCollector:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.act_stats = {}

    def register_hooks(self):
        def make_hook(name):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                act = act.detach().float()
                if name not in self.act_stats:
                    self.act_stats[name] = {'min': [], 'max': []}
                act_min = act.amin(dim=list(range(1, act.dim())), keepdim=True)
                act_max = act.amax(dim=list(range(1, act.dim())), keepdim=True)
                self.act_stats[name]['min'].append(act_min.cpu())
                self.act_stats[name]['max'].append(act_max.cpu())
            return hook_fn

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(make_hook(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def collect(self, latents, num_samples=20):
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(min(num_samples, len(latents))), desc="Calibration forward"):
                x = latents[i: i + 1].cuda().to(torch.bfloat16)  # ← must be bfloat16
                t = torch.tensor([1000.0], device="cuda", dtype=torch.bfloat16)
                context = torch.randn(1, 10, 4096, device="cuda", dtype=torch.bfloat16)
                try:
                    self.model(x, t, context)
                except Exception as e:
                    print(f"  Warning: pass {i} error: {e}")

    def compute_scales(self):
        scales = {}
        for name, stats in self.act_stats.items():
            if not stats['min']:
                continue
            all_min = torch.cat(stats['min'], dim=0)
            all_max = torch.cat(stats['max'], dim=0)
            act_min = torch.amin(all_min, dim=0)
            act_max = torch.amax(all_max, dim=0)
            act_range = act_max - act_min
            act_scale = act_range / 255.0
            act_scale = torch.clamp(act_scale, min=1e-6)
            zero_pt = torch.round(-act_min / act_scale)
            scales[name] = {
                'act_min': act_min, 'act_max': act_max,
                'act_scale': act_scale, 'zero_point': zero_pt,
            }
        return scales


def sample_latents_from_video(video_path, num_latents=32, latent_size=(60, 80)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while len(frames) < num_latents:
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, total))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (latent_size[1], latent_size[0]))
            frame = frame.astype(np.float32) / 255.0
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
    cap.release()
    latents = np.zeros((len(frames), 16, latent_size[0], latent_size[1]), dtype=np.float32)
    for i, frame in enumerate(frames):
        for c in range(16):
            latents[i, c] = frame[c % 3]
    return torch.from_numpy(latents)  # float32


def load_model_bf16(checkpoint_path):
    """Load WanModel in bfloat16 on GPU, then load weights."""
    from safetensors.torch import load_file
    sd = load_file(checkpoint_path)
    new_sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}

    # Init in bf16 on GPU FIRST (before loading weights), then load
    model = WanModel(
        dim=1536, eps=1e-5, ffn_dim=8960, freq_dim=256, in_dim=16,
        num_heads=12, num_layers=30, out_dim=16, patch_size=(1, 2, 2), text_dim=4096,
    )
    model = model.to("cuda").to(torch.bfloat16)
    model.load_state_dict(new_sd, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ckpt", required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--video", default="data/lowres/carphone_qcif.mp4")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--latent_size", default="60x80")
    args = parser.parse_args()

    latent_h, latent_w = map(int, args.latent_size.split("x"))

    print(f"Loading model (bfloat16 on CUDA)...")
    model = load_model_bf16(args.input_ckpt)

    print(f"Sampling {args.num_samples} latents from {args.video}...")
    latents = sample_latents_from_video(
        args.video, num_latents=args.num_samples,
        latent_size=(latent_h, latent_w)
    )
    print(f"  Latents: {latents.shape}, {latents.dtype}")

    collector = ActivationCollector(model)
    collector.register_hooks()
    collector.collect(latents, num_samples=args.num_samples)

    print("Computing scales...")
    scales = collector.compute_scales()
    collector.remove_hooks()

    cache_data = {}
    for name, data in scales.items():
        cache_data[name] = {
            'act_min': data['act_min'].numpy().tolist(),
            'act_max': data['act_max'].numpy().tolist(),
            'act_scale': data['act_scale'].numpy().tolist(),
            'zero_point': data['zero_point'].numpy().tolist(),
        }

    os.makedirs(os.path.dirname(args.output_cache) or ".", exist_ok=True)
    with open(args.output_cache, "w") as f:
        json.dump(cache_data, f, indent=2)
    print(f"Saved {len(cache_data)}-layer calibration cache → {args.output_cache}")


if __name__ == "__main__":
    main()
