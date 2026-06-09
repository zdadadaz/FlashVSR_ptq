"""PTQ W8A8 calibrator for FlashVSR.

Collects activation statistics (min/max) for per-tensor quantization
using TensorRT-compatible W8A8 calibration.
"""

import sys
from pathlib import Path

# Add project root to path for src imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.wan_video_dit import WanModel, sinusoidal_embedding_1d


@dataclass
class CalibrationSample:
    """Single calibration sample with latent-simulated input."""

    latents: torch.Tensor  # (T, 16, H, W), bf16
    timesteps: torch.Tensor  # (1,), int64
    contexts: torch.Tensor  # (10, 4096), bf16


class FlashVSRTQDataset(Dataset):
    """Dataset for PTQ calibration using DOVE frames."""

    TIMESTEPS = [0, 200, 400, 600, 800, 999]

    def __init__(
        self,
        root: str = "datasets",
        num_samples: int = 320,
        frame_size: tuple = (64, 64),  # Must be divisible by 16 for window partitioning
        num_frames: int = 6,  # Must be divisible by 2 for temporal window partitioning (win[0]=2)
    ):
        self.root = Path(root)
        self.num_samples = num_samples
        self.frame_size = frame_size
        self.num_frames = num_frames

        # Try DOVE train set first, fallback to test/UDM10/GT
        self.dove_path = self.root / "train" / "HQ-VSR"
        self.fallback_path = self.root / "test" / "UDM10" / "GT"

        if self.dove_path.exists():
            self.frames = sorted(self.dove_path.rglob("*.png"))
        elif self.fallback_path.exists():
            self.frames = sorted(self.fallback_path.rglob("*.png"))
        else:
            self.frames = []

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> CalibrationSample:
        if self.frames:
            # Load random frame
            frame_path = random.choice(self.frames)
            img = torch.from_numpy(
                torch.load(frame_path, map_location="cpu").numpy()
                if frame_path.suffix == ".pt"
                else self._load_image(frame_path)
            )
            # Resize to frame_size
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.frame_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            # Fallback: return zero latent
            img = torch.zeros(3, *self.frame_size)

        # Replicate RGB to 16 channels (simulate 16ch latent)
        latent = img.repeat(16 // 3 + 1, 1, 1)[:16, :, :].to(
            dtype=torch.bfloat16
        )

        # Stack num_frames copies for temporal dimension.
        # Output: (num_frames, C, H, W), matching run_calibration's
        # downstream (B, T, C, H, W) batching contract.
        latents = latent.unsqueeze(0).expand(self.num_frames, -1, -1, -1).contiguous()

        return CalibrationSample(
            latents=latents,  # (num_frames, C, H, W)
            timesteps=torch.tensor(
                [random.choice(self.TIMESTEPS)], dtype=torch.int64
            ),
            contexts=torch.randn(10, 4096, dtype=torch.bfloat16),
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image from path using PIL."""
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img


class ActivationCollector:
    """Collects activation min/max statistics for quantization calibration."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.act_stats: Dict[str, Dict[str, List[float]]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self) -> None:
        """Register forward hooks on all Linear, Conv2d, Conv3d layers."""

        def forward_hook(name: str):
            def hook_fn(module, input, output):
                tensor = output if isinstance(output, torch.Tensor) else output[0]
                # Collect min/max over spatial dims (dim 1 onward)
                amin = tensor.amin(dim=list(range(1, tensor.ndim))).float()
                amax = tensor.amax(dim=list(range(1, tensor.ndim))).float()

                if name not in self.act_stats:
                    self.act_stats[name] = {"min": [], "max": []}
                self.act_stats[name]["min"].append(amin)
                self.act_stats[name]["max"].append(amax)

            return hook_fn

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                handle = module.register_forward_hook(forward_hook(name))
                self._handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def compute_scales(self) -> Dict[str, Dict[str, float]]:
        """Compute per-tensor quantization scales from collected stats.

        Returns:
            Dict mapping layer name to {'act_min', 'act_max', 'act_scale', 'zero_point'}
        """
        scales = {}
        for name, stats in self.act_stats.items():
            all_min = torch.cat(stats["min"])
            all_max = torch.cat(stats["max"])

            act_min = all_min.min().item()
            act_max = all_max.max().item()
            act_scale = (act_max - act_min) / 255.0
            zero_point = round(-act_min / act_scale)

            scales[name] = {
                "act_min": act_min,
                "act_max": act_max,
                "act_scale": act_scale,
                "zero_point": zero_point,
            }

        return scales


def collate_calibration_samples(batch):
    """Collate function to stack CalibrationSample dataclasses into batched tensors."""
    return {
        "latents": torch.stack([s.latents for s in batch]),
        "timesteps": torch.stack([s.timesteps for s in batch]),
        "contexts": torch.stack([s.contexts for s in batch]),
    }


def run_calibration(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, Dict[str, float]]:
    """Run calibration on model using dataset.

    Args:
        model: Model to calibrate
        dataset: Calibration dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers

    Returns:
        Dict of scales per layer
    """
    collector = ActivationCollector(model)
    collector.register_hooks()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_calibration_samples,
    )

    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in tqdm(loader, desc="Calibration"):
            # Dataset returns (B, T, C, H, W) where T=num_frames
            latents = batch["latents"].to(device=device, dtype=dtype)  # (B, T, C, H, W)
            timesteps = batch["timesteps"].to(device).float().squeeze(-1)  # (B,)
            contexts = batch["contexts"].to(device=device, dtype=dtype)  # (B, 10, 4096)

            try:
                for i in range(latents.shape[0]):
                    # Use model_fn_wan_video pattern (same as pipeline)
                    # Reshape from (T, C, H, W) to (C, T, H, W) for model
                    x = latents[i].permute(1, 0, 2, 3)  # (C, T, H, W)
                    x = x.unsqueeze(0)  # (1, C, T, H, W)

                    # Precompute timestep embedding (same as pipeline does)
                    t = model.time_embedding(
                        sinusoidal_embedding_1d(model.freq_dim, timesteps[i:i+1])
                    )
                    t_mod = model.time_projection(t).unflatten(1, (6, model.dim))

                    # Patchify: x->(B, seq, C), grid_size->(f, h, w)
                    x, (f, h, w) = model.patchify(x)

                    # RoPE frequencies
                    win = (2, 8, 8)
                    freqs = torch.cat([
                        model.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                        model.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        model.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
                    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

                    seqlen = f // win[0]
                    local_num = seqlen
                    window_size = win[0] * h * w // 128
                    square_num = window_size * window_size
                    topk = int(square_num * 2.0) - 1
                    kv_len = int(3.0)

                    # Forward through blocks
                    for block_id, block in enumerate(model.blocks):
                        try:
                            block_out = block(
                                x, contexts[i:i+1], t_mod, freqs, f, h, w,
                                local_num, topk,
                                block_id=block_id,
                                kv_len=kv_len,
                                is_full_block=False,
                                is_stream=True,  # Required for 3-value return
                                pre_cache_k=None,
                                pre_cache_v=None,
                                local_range=9,
                            )
                            if isinstance(block_out, tuple):
                                x, _, _ = block_out
                            else:
                                x = block_out
                        except ValueError:
                            # Fallback: is_stream=False path (model bug - always expects 3 returns)
                            block_out = block(
                                x, contexts[i:i+1], t_mod, freqs, f, h, w,
                                local_num, topk,
                                block_id=block_id,
                                kv_len=kv_len,
                                is_full_block=False,
                                is_stream=False,
                                pre_cache_k=None,
                                pre_cache_v=None,
                                local_range=9,
                            )
                            if isinstance(block_out, tuple):
                                x, _, _ = block_out
                            else:
                                x = block_out
            except Exception as e:
                print(f"  Warning: forward pass error: {e}")
                continue

    scales = collector.compute_scales()
    collector.remove_hooks()

    return scales


def save_calibration_cache(
    scales: Dict[str, Dict[str, float]], output_path: str
) -> None:
    """Save calibration cache to JSON file.

    Args:
        scales: Computed scales dict
        output_path: Path to save JSON
    """
    # Convert to JSON-serializable format
    cache = {}
    for name, stats in scales.items():
        cache[name] = {k: float(v) for k, v in stats.items()}

    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)


def main() -> None:
    """Main entry point for calibration."""
    parser = argparse.ArgumentParser(description="PTQ W8A8 calibration")
    parser.add_argument(
        "--input_ckpt", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--output_cache", type=str, required=True, help="Output cache path"
    )
    parser.add_argument(
        "--dataset", type=str, default="datasets", help="Dataset root"
    )
    parser.add_argument(
        "--samples", type=int, default=320, help="Number of calibration samples"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    args = parser.parse_args()

    # Load model from checkpoint
    print(f"Loading model from {args.input_ckpt}")
    if args.input_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(args.input_ckpt)
    else:
        state_dict = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)

    # Create model instance with standard FlashVSR config
    model = WanModel(
        dim=1536,
        eps=1e-5,
        ffn_dim=8960,
        freq_dim=256,
        in_dim=16,
        num_heads=12,
        num_layers=30,
        out_dim=16,
        patch_size=(1, 2, 2),
        text_dim=4096
    )

    # Handle state_dict keys if they have 'model.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Move model to CUDA for calibration
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")

        # Initialize cross-attention KV cache with a dummy context
        # This is required for the model's cross-attention layers to work
        dummy_context = torch.randn(1, 10, 4096, dtype=torch.float32, device='cuda')
        if hasattr(model, 'reinit_cross_kv'):
            model.reinit_cross_kv(dummy_context)
            print("Cross-attention KV cache initialized")
    else:
        print("WARNING: CUDA not available, running on CPU")

    # Create dataset
    dataset = FlashVSRTQDataset(
        root=args.dataset,
        num_samples=args.samples,
        frame_size=(64, 64),  # Must be divisible by 16 for window partitioning
    )
    print(f"Created dataset with {len(dataset)} samples")

    # Run calibration
    print("Running calibration...")
    scales = run_calibration(
        model,
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Save cache
    save_calibration_cache(scales, args.output_cache)
    print(f"Saved calibration cache to {args.output_cache}")


if __name__ == "__main__":
    main()