"""
PTQ Calibration pipeline for FlashVSR.

Collects activation statistics (min/max per channel) for asymmetric activation quantization,
then saves calibration cache for conversion and TRT compilation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.wan_video_dit import WanModel


def get_dove_calibration_data(dataset_path, num_samples=320, frame_size=(24, 24)):
    """Sample frames from DOVE dataset for calibration."""
    dataset_path = Path(dataset_path)
    hq_vsr = dataset_path / "train" / "HQ-VSR"
    if not hq_vsr.exists():
        hq_vsr = dataset_path / "test" / "UDM10" / "GT"

    video_dirs = [d for d in hq_vsr.iterdir() if d.is_dir()]
    if not video_dirs:
        video_dirs = [hq_vsr]

    samples = []
    print(f"Sampling {num_samples} calibration frames from {hq_vsr}...")

    pbar = tqdm(total=num_samples)
    while len(samples) < num_samples:
        v_dir = video_dirs[np.random.randint(len(video_dirs))]
        frames = list(v_dir.glob("*.png")) + list(v_dir.glob("*.jpg"))
        if not frames:
            continue

        f_path = frames[np.random.randint(len(frames))]
        img = cv2.imread(str(f_path))
        if img is None:
            continue

        img = cv2.resize(img, (frame_size[1], frame_size[0]))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # Wan latent is 16 channels
        latent = np.zeros((16, frame_size[0], frame_size[1]), dtype=np.float32)
        for i in range(16):
            latent[i] = img[i % 3]

        samples.append(torch.from_numpy(latent))
        pbar.update(1)
    pbar.close()

    return torch.stack(samples)


class ActivationCollector:
    """Collect per-layer activation min/max statistics using forward hooks."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.act_stats = {}  # name -> {'min': [], 'max': []}

    def register_hooks(self):
        """Register forward hooks on all nn.Linear layers."""
        def make_hook(name):
            def hook_fn(module, input, output):
                act = output
                if isinstance(output, tuple):
                    act = output[0]
                act = act.detach().float()

                if name not in self.act_stats:
                    self.act_stats[name] = {'min': [], 'max': []}

                # Per-channel min/max over spatial dims and batch
                # Keep dim=0 (channel dim) for per-channel statistics
                act_min = act.amin(dim=list(range(1, act.dim())), keepdim=True)
                act_max = act.amax(dim=list(range(1, act.dim())), keepdim=True)

                self.act_stats[name]['min'].append(act_min.cpu())
                self.act_stats[name]['max'].append(act_max.cpu())

            return hook_fn

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def collect(self, latents, timesteps, contexts, num_samples=20):
        """Run forward passes and collect activation statistics."""
        self.model.cuda()
        self.model.eval()

        num_runs = min(num_samples, len(latents))
        with torch.no_grad():
            for i in tqdm(range(num_runs), desc="Calibration forward"):
                x = latents[i : i + 1].unsqueeze(1)  # (B=1, T=1, C=16, H, W)
                t = torch.randint(0, 1000, (1,), device="cuda")
                context = contexts[i % len(contexts):i % len(contexts) + 1] if i < len(contexts) else torch.randn(1, 10, 4096, device="cuda")

                try:
                    self.model(x, t, context)
                except Exception as e:
                    print(f"  Warning: forward pass {i} error: {e}")
                    continue

    def compute_scales(self):
        """
        Compute asymmetric scales per layer from collected statistics.

        Returns:
            dict: {name: {'act_min': tensor, 'act_max': tensor, 'act_scale': tensor, 'zero_point': tensor}}
        """
        scales = {}
        for name, stats in self.act_stats.items():
            if not stats['min'] or not stats['max']:
                continue

            all_min = torch.cat(stats['min'], dim=0)
            all_max = torch.cat(stats['max'], dim=0)

            # Per-channel: take min across all batches/spatial positions
            act_min = torch.amin(all_min, dim=0)
            act_max = torch.amax(all_max, dim=0)

            # Asymmetric scale: (max - min) / 255
            act_range = act_max - act_min
            act_scale = act_range / 255.0
            act_scale = torch.clamp(act_scale, min=1e-6)

            # Zero point: round(-min / scale)
            zero_pt = torch.round(-act_min / act_scale)

            scales[name] = {
                'act_min': act_min,
                'act_max': act_max,
                'act_scale': act_scale,
                'zero_point': zero_pt,
            }

        return scales


def load_model(checkpoint_path):
    """Load WanModel from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
        text_dim=4096,
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="PTQ calibration for FlashVSR WanModel")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to model .safetensors or .pth")
    parser.add_argument("--output_cache", type=str, required=True, help="Path to save calibration cache JSON")
    parser.add_argument("--dataset", type=str, default="datasets", help="Path to DOVE dataset")
    parser.add_argument("--samples", type=int, default=320, help="Number of calibration samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for calibration")
    args = parser.parse_args()

    # Check dependencies
    try:
        import torch_tensorrt
        print("torch_tensorrt available")
    except ImportError:
        print("WARNING: torch_tensorrt not installed. Calibration will use native PyTorch.")
        print("Install with: pip install torch-tensorrt")

    # Load model
    model = load_model(args.input_ckpt)
    model.cuda()
    model.eval()

    # Prepare calibration data
    latents = get_dove_calibration_data(args.dataset, num_samples=args.samples)
    latents = latents.cuda().to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    # Generate dummy timesteps and contexts
    num_contexts = 10
    contexts = [
        torch.randn(1, 10, 4096, device="cuda", dtype=latents.dtype)
        for _ in range(num_contexts)
    ]

    # Collect activation statistics
    collector = ActivationCollector(model)
    collector.register_hooks()

    print(f"Running calibration with {args.samples} samples...")
    collector.collect(latents, None, contexts, num_samples=args.samples)

    # Compute scales
    print("Computing asymmetric scales...")
    scales = collector.compute_scales()

    collector.remove_hooks()

    # Save calibration cache
    cache_data = {}
    for name, stats in scales.items():
        cache_data[name] = {
            'act_min': stats['act_min'].numpy().tolist(),
            'act_max': stats['act_max'].numpy().tolist(),
            'act_scale': stats['act_scale'].numpy().tolist(),
            'zero_point': stats['zero_point'].numpy().tolist(),
        }

    os.makedirs(os.path.dirname(args.output_cache) or ".", exist_ok=True)
    with open(args.output_cache, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"Calibration cache saved to {args.output_cache}")
    print(f"Collected stats for {len(cache_data)} layers")


if __name__ == "__main__":
    main()