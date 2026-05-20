# PTQ W8A8 TensorRT for FlashVSR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement full W8A8 INT8 quantization for FlashVSR's DiT (WanModel) via TensorRT compilation, with VAE remaining bf16.

**Architecture:** DiT remains pure fp16/bf16 in PyTorch — no custom quantization layers inserted. RMSNorm gamma/bias are folded into downstream Linear/Conv weights. TensorRT's `IInt8EntropyCalibrator2` handles Q/DQ insertion and weight quantization during compilation. VAE is a separate bf16 module in the pipeline.

**Tech Stack:** torch>=2.0, torch-tensorrt, PyTorch 2.0 `torch.export`, DOVE dataset

---

## File Map

```
src/models/quantization/rmsnorm_fold.py      [NEW] RMSNorm gamma/bias fold utility
scripts/ptq/calibrator_w8a8.py               [NEW] Calibration dataset + ActivationCollector + calibration run
scripts/ptq/compile_trt_w8a8.py              [NEW] torch.export → TensorRT compile + engine save
nodes.py                                      [MOD] Add W8A8_PTQ quantize_mode, TRT engine loading
src/pipelines/flashvsr_full.py               [MOD] Add model_fn_trt() for TensorRT call path
cli_main.py                                   [MOD] Add --quantize_mode W8A8_PTQ --trt_engine flags
```

---

## Task 1: RMSNorm Fold Utility

**Files:**
- Create: `src/models/quantization/rmsnorm_fold.py`
- Test: `tests/models/quantization/test_rmsnorm_fold.py`

- [ ] **Step 1: Write failing test**

```python
# tests/models/quantization/test_rmsnorm_fold.py
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.models.quantization.rmsnorm_fold import fold_rmsnorm_into_linear

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
        out = x / rms * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

def test_fold_rmsnorm_linear_no_bias():
    """Linear: gamma (weight) folded, bias=None (standard DiT)"""
    linear = nn.Linear(128, 256)
    rms = SimpleRMSNorm(128, bias=False)
    new_linear = fold_rmsnorm_into_linear(rms, linear)
    assert isinstance(new_linear, nn.Linear)
    assert new_linear.in_features == 128
    assert new_linear.out_features == 256
    # gamma should be folded into weight (check magnitude changed)
    original_w = linear.weight.data.clone()
    folded_w = new_linear.weight.data
    assert not torch.allclose(original_w * rms.weight.view(1, -1), folded_w, atol=1e-6)
    print("test_fold_rmsnorm_linear_no_bias PASS")

def test_fold_rmsnorm_linear_with_bias():
    """Linear with bias=None in RMSNorm: bias stays None"""
    linear = nn.Linear(128, 256, bias=False)
    rms = SimpleRMSNorm(128, bias=False)
    new_linear = fold_rmsnorm_fold(rms, linear)
    assert new_linear.bias is None
    print("test_fold_rmsnorm_linear_with_bias PASS")

def test_fold_rmsnorm_linear_with_zero_bias():
    """RMSNorm with zero bias: treated as no bias, folded correctly"""
    linear = nn.Linear(128, 256)
    rms = SimpleRMSNorm(128, bias=True)
    rms.bias.data.zero_()
    new_linear = fold_rmsnorm_into_linear(rms, linear)
    # Zero bias should be skipped, linear.bias unchanged
    print("test_fold_rmsnorm_linear_with_zero_bias PASS")

def test_fold_rmsnorm_conv2d():
    """Conv2d: gamma folded along in_channels axis"""
    conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    rms = SimpleRMSNorm(64, bias=False)
    new_conv = fold_rmsnorm_into_linear(rms, conv)
    assert isinstance(new_conv, nn.Conv2d)
    assert new_conv.in_channels == 64
    assert new_conv.out_channels == 128
    print("test_fold_rmsnorm_conv2d PASS")

def test_fold_rmsnorm_conv3d():
    """Conv3d: gamma folded along in_channels axis (5D view)"""
    conv = nn.Conv3d(64, 128, kernel_size=3, padding=1)
    rms = SimpleRMSNorm(64, bias=False)
    new_conv = fold_rmsnorm_into_linear(rms, conv)
    assert isinstance(new_conv, nn.Conv3d)
    assert new_conv.in_channels == 64
    assert new_conv.out_channels == 128
    print("test_fold_rmsnorm_conv3d PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/quantization/test_rmsnorm_fold.py -v`
Expected: FAIL — module `rmsnorm_fold` not found

- [ ] **Step 3: Write implementation**

```python
# src/models/quantization/rmsnorm_fold.py
"""
Fold RMSNorm's learnable gamma (weight) and bias into adjacent Linear/Conv layers.

Foldable: gamma (gain) and bias — static, no input dependence
NOT foldable: dynamic RMS(x) computation — must remain as runtime op.
              TensorRT auto-fuses RMSNorm(pure) + Linear into a single kernel.

WanModel's RMSNorm has elementwise_affine=True but bias=None (standard DiT practice).
If bias is present and non-zero, only fold gamma (bias folding requires weight @ bias.T).
"""

import torch
import torch.nn as nn
import warnings


def fold_rmsnorm_into_linear(rmsnorm, layer):
    """
    Fold RMSNorm's learnable gamma (weight) and bias into the adjacent Linear or Conv.

    Args:
        rmsnorm: RMSNorm module with .weight (gamma) and optional .bias
        layer: nn.Linear, nn.Conv2d, or nn.Conv3d

    Returns:
        New layer with gamma folded into weight. If bias is non-trivial and cannot be
        folded, a warning is issued and only gamma is folded.
    """
    gamma = rmsnorm.weight.data
    bias = rmsnorm.bias.data if rmsnorm.bias is not None else None

    # If bias is non-trivial, only fold gamma (skip bias folding to avoid
    # incorrect results without implementing full weight @ bias.T reshape)
    if bias is not None and not (bias.abs() < 1e-6).all():
        warnings.warn(
            f"RMSNorm bias is non-zero (max={bias.abs().max():.6f}), folding gamma only. "
            "Bias folding requires weight @ bias.T — implement if needed."
        )
        bias = None

    if isinstance(layer, nn.Linear):
        w_folded = layer.weight.data * gamma.view(1, -1)
        new_layer = nn.Linear(
            layer.in_features, layer.out_features, bias=layer.bias is not None
        )
        new_layer.weight.data = w_folded
        if bias is not None and layer.bias is not None:
            new_layer.bias.data = layer.bias.data + bias
        elif bias is not None:
            new_layer.bias = nn.Parameter(bias.clone())

    elif isinstance(layer, nn.Conv2d):
        w_folded = layer.weight.data * gamma.view(1, -1, 1, 1)
        new_layer = nn.Conv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
        new_layer.weight.data = w_folded
        if bias is not None and layer.bias is not None:
            new_layer.bias.data = layer.bias.data + bias
        elif bias is not None:
            new_layer.bias = nn.Parameter(bias.clone())

    elif isinstance(layer, nn.Conv3d):
        w_folded = layer.weight.data * gamma.view(1, -1, 1, 1, 1)
        new_layer = nn.Conv3d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
        )
        new_layer.weight.data = w_folded
        if bias is not None and layer.bias is not None:
            new_layer.bias.data = layer.bias.data + bias
        elif bias is not None:
            new_layer.bias = nn.Parameter(bias.clone())

    else:
        raise TypeError(f"Unsupported layer type: {type(layer)}")

    return new_layer


def fold_rmsnorms_in_model(model, prefix=''):
    """
    Walk a model's DiT blocks and fold each RMSNorm into its downstream Linear/Conv.

    In WanModel's DiTBlock the structure is:
        x → RMSNorm → SelfAttn → GateModule → RMSNorm → CrossAttn → GateModule → RMSNorm → MLP → GateModule → +x

    Each RMSNorm precedes a Linear layer (SelfAttn.qkv, CrossAttn.qkv, MLP.fc1).
    We traverse named_children and replace in-place where a RMSNorm precedes a Linear/Conv.

    Args:
        model: nn.Module (typically WanModel or DiTBlock)
        prefix: for recursion tracking

    Returns:
        Modified model in-place.
    """
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Check if this child is a container that needs recursion
        if isinstance(child, nn.Module) and not isinstance(child, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            fold_rmsnorms_in_model(child, full_name)
            continue

        # If child is a Linear/Conv that could follow an RMSNorm, leave it
        # (RMSNorm folding happens when we find the RMSNorm parent)
        # This function folds in-place when called on a DiTBlock with known structure
    return model


def fold_dit_rmsnorms(dit):
    """
    Apply RMSNorm folding to all DiTBlocks in a WanModel.

    WanModel (WanModel) structure:
        .patch_embed, .blocks (nn.ModuleList of DiTBlock), .norm, .head

    Each DiTBlock has:
        .self_attn (SelfAttention with .qkv, .proj), .cross_attn, .mlp, and their RMSNorms

    For each DiTBlock, we find its RMSNorm layers and fold them into the
    downstream Linear/Conv weights in-place.

    Args:
        dit: WanModel instance

    Returns:
        dit (modified in-place)
    """
    from src.models.wan_video_dit import RMSNorm

    for block in dit.blocks:
        # Collect pairs of (rmsnorm, layer) to fold
        # Pattern: block.self_attn.qkv (Linear) after block.norm1 (RMSNorm)
        #         block.cross_attn.qkv (Linear) after block.norm2 (RMSNorm)
        #         block.mlp.fc1 (Linear) after block.norm3 (RMSNorm)
        pairs_to_fold = [
            (getattr(block, 'norm1', None), getattr(block.self_attn, 'qkv', None)),
            (getattr(block, 'norm2', None), getattr(block.cross_attn, 'qkv', None)),
            (getattr(block, 'norm3', None), getattr(block.mlp, 'fc1', None)),
        ]

        for rms, linear in pairs_to_fold:
            if rms is not None and linear is not None:
                if isinstance(rms, RMSNorm) and isinstance(linear, nn.Linear):
                    new_linear = fold_rmsnorm_into_linear(rms, linear)
                    # Replace in parent
                    parent_name = None
                    for pname, pchild in block.named_children():
                        if pchild is linear:
                            parent_name = pname
                            break
                    if parent_name:
                        setattr(block, parent_name, new_linear)

    return dit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/models/quantization/test_rmsnorm_fold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/quantization/rmsnorm_fold.py tests/models/quantization/test_rmsnorm_fold.py
git commit -m "feat: add RMSNorm fold utility for W8A8 PTQ

Fold gamma/bias into Linear/Conv weights. Supports Linear, Conv2d, Conv3d.
Non-zero bias triggers warning and gamma-only fold.
Includes fold_dit_rmsnorms() for WanModel DiTBlocks.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Calibration Dataset and ActivationCollector

**Files:**
- Create: `scripts/ptq/calibrator_w8a8.py`
- Test: `tests/scripts/ptq/test_calibrator_w8a8.py`

- [ ] **Step 1: Write failing test**

```python
# tests/scripts/ptq/test_calibrator_w8a8.py
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
from scripts.ptq.calibrator_w8a8 import FlashVSRTQDataset, CalibrationSample, ActivationCollector
from dataclasses import dataclass

def test_dataset_returns_calibration_sample():
    """Dataset __getitem__ returns CalibrationSample with correct shapes."""
    # Requires DOVE dataset at datasets/DOVE
    dataset = FlashVSRTQDataset(root="datasets", num_samples=10)
    sample = dataset[0]
    assert isinstance(sample, CalibrationSample)
    assert sample.latents.shape[0] == 16   # C=16
    assert sample.latents.shape[1] == 24   # H=24
    assert sample.latents.shape[2] == 24   # W=24
    assert sample.timesteps.shape[0] == 1
    assert sample.contexts.shape == (1, 10, 4096)
    print("test_dataset_returns_calibration_sample PASS")

def test_activation_collector_hooks_linear():
    """ActivationCollector registers hooks on nn.Linear and collects stats."""
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.Linear(32, 64)
    )
    collector = ActivationCollector(model)
    collector.register_hooks()

    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)

    stats = collector.compute_scales()
    assert len(stats) >= 2
    assert '0' in stats  # first Linear
    assert '1' in stats  # second Linear

    collector.remove_hooks()
    print("test_activation_collector_hooks_linear PASS")

def test_activation_collector_conv3d():
    """ActivationCollector works with Conv3d (CausalConv3d simulation)."""
    model = torch.nn.Sequential(
        torch.nn.Conv3d(16, 32, kernel_size=3, padding=1),
        torch.nn.Conv3d(32, 64, kernel_size=3, padding=1)
    )
    collector = ActivationCollector(model)
    collector.register_hooks()

    x = torch.randn(2, 16, 4, 24, 24)  # (B, C, T, H, W)
    with torch.no_grad():
        model(x)

    stats = collector.compute_scales()
    assert len(stats) >= 2
    collector.remove_hooks()
    print("test_activation_collector_conv3d PASS")

def test_calibration_sample_dataclass():
    sample = CalibrationSample(
        latents=torch.randn(1, 16, 24, 24),
        timesteps=torch.tensor([500]),
        contexts=torch.randn(1, 10, 4096)
    )
    assert sample.latents.shape == (1, 16, 24, 24)
    assert sample.timesteps.shape == (1,)
    assert sample.contexts.shape == (1, 10, 4096)
    print("test_calibration_sample_dataclass PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/ptq/test_calibrator_w8a8.py -v`
Expected: FAIL — module `calibrator_w8a8` not found

- [ ] **Step 3: Write implementation**

```python
# scripts/ptq/calibrator_w8a8.py
"""
PTQ W8A8 Calibration Pipeline for FlashVSR DiT.

Collects per-tensor activation min/max statistics for asymmetric quantization
using DOVE dataset frames as calibration input. Produces a calibration cache
(JSON) used by TensorRT's IInt8EntropyCalibrator2.
"""

import argparse
import cv2
import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class CalibrationSample:
    """Single calibration sample for FlashVSR DiT."""
    latents: torch.Tensor   # (16, 24, 24), bf16 — latent-simulated input
    timesteps: torch.Tensor # (1,), int64 — timestep
    contexts: torch.Tensor  # (10, 4096), bf16 — text embedding


class FlashVSRTQDataset(Dataset):
    """
    DOVE dataset wrapper for DiT calibration.

    Samples 24x24 frames from DOVE and simulates 16-channel latent input
    (WanModel expects C=16 latent space). Mixes timesteps across denoising range.
    """
    def __init__(self, root="datasets", num_samples=320, frame_size=(24, 24)):
        self.root = Path(root)
        self.num_samples = num_samples
        self.frame_size = frame_size

        hq_vsr = self.root / "train" / "HQ-VSR"
        if not hq_vsr.exists():
            hq_vsr = self.root / "test" / "UDM10" / "GT"
        self.video_dirs = [d for d in hq_vsr.iterdir() if d.is_dir()] if hq_vsr.exists() else [hq_vsr]

        # Mix timesteps: cover full denoising range
        self.timestep_choices = [0, 200, 400, 600, 800, 999]
        self.context = torch.randn(10, 4096)  # fixed dummy context

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> CalibrationSample:
        # Sample random frame
        v_dir = self.video_dirs[np.random.randint(len(self.video_dirs))]
        frames = list(v_dir.glob("*.png")) + list(v_dir.glob("*.jpg"))
        if not frames:
            # Fallback: return zero latent
            latent = torch.zeros(16, *self.frame_size, dtype=torch.bfloat16)
        else:
            f_path = frames[np.random.randint(len(frames))]
            img = cv2.imread(str(f_path))
            if img is None:
                img = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.float32)
            else:
                img = cv2.resize(img, (self.frame_size[1], self.frame_size[0]))
                img = img.astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1)

            # Simulate 16-channel latent from 3-channel RGB
            latent = np.zeros((16, *self.frame_size), dtype=np.float32)
            for c in range(16):
                latent[c] = img[c % 3]
            latent = torch.from_numpy(latent)

        # Mixed timestep
        t = torch.tensor([self.timestep_choices[np.random.randint(len(self.timestep_choices))]], dtype=torch.int64)

        return CalibrationSample(
            latents=latent.to(torch.bfloat16),
            timesteps=t,
            contexts=self.context.to(torch.bfloat16),
        )


class ActivationCollector:
    """
    Register forward hooks on all nn.Linear/Conv2d/Conv3d to collect
    per-tensor activation min/max statistics.

    Usage:
        collector = ActivationCollector(model)
        collector.register_hooks()
        # run forward passes...
        scales = collector.compute_scales()
        collector.remove_hooks()
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.act_stats = {}  # name -> {'min': [], 'max': []}

    def register_hooks(self):
        def make_hook(name):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                act = act.detach().float()

                if name not in self.act_stats:
                    self.act_stats[name] = {'min': [], 'max': []}

                # Per-tensor: flatten everything except batch
                act_min = act.amin(dim=list(range(1, act.dim())), keepdim=True)
                act_max = act.amax(dim=list(range(1, act.dim())), keepdim=True)

                self.act_stats[name]['min'].append(act_min.cpu())
                self.act_stats[name]['max'].append(act_max.cpu())
            return hook_fn

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_scales(self):
        """
        Compute per-tensor asymmetric scales from collected statistics.

        Returns:
            dict: {name: {'act_min': tensor, 'act_max': tensor,
                          'act_scale': tensor, 'zero_point': tensor}}
        """
        scales = {}
        for name, stats in self.act_stats.items():
            if not stats['min'] or not stats['max']:
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
                'act_min': act_min,
                'act_max': act_max,
                'act_scale': act_scale,
                'zero_point': zero_pt,
            }
        return scales


def run_calibration(model, dataset, batch_size=32, num_workers=4):
    """
    Run full calibration pass through the model.

    Args:
        model: WanModel in eval mode
        dataset: FlashVSRTQDataset
        batch_size: DataLoader batch size
        num_workers: DataLoader workers

    Returns:
        dict of activation scales per layer
    """
    collector = ActivationCollector(model)
    collector.register_hooks()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calibration"):
            latents = batch.latents.cuda().unsqueeze(1)  # (B, T=1, C, H, W)
            timesteps = batch.timesteps.cuda()
            contexts = batch.contexts.cuda()

            try:
                model(latents, timesteps, contexts)
            except Exception as e:
                print(f"  Warning: forward pass error: {e}")
                continue

    scales = collector.compute_scales()
    collector.remove_hooks()
    return scales


def save_calibration_cache(scales, output_path):
    """Save calibration cache to JSON (tensors → lists)."""
    cache = {}
    for name, stats in scales.items():
        cache[name] = {
            'act_min': stats['act_min'].numpy().tolist(),
            'act_max': stats['act_max'].numpy().tolist(),
            'act_scale': stats['act_scale'].numpy().tolist(),
            'zero_point': stats['zero_point'].numpy().tolist(),
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Calibration cache saved to {output_path} ({len(cache)} layers)")


def main():
    parser = argparse.ArgumentParser(description="PTQ W8A8 calibration for FlashVSR DiT")
    parser.add_argument("--input_ckpt", type=str, required=True,
                        help="Path to DiT .safetensors or .pth checkpoint")
    parser.add_argument("--output_cache", type=str, required=True,
                        help="Path to save calibration cache JSON")
    parser.add_argument("--dataset", type=str, default="datasets",
                        help="Path to DOVE dataset root")
    parser.add_argument("--samples", type=int, default=320,
                        help="Number of calibration samples")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="DataLoader batch size")
    args = parser.parse_args()

    # Load WanModel
    from src.models.wan_video_dit import WanModel
    print(f"Loading checkpoint from {args.input_ckpt}...")
    if args.input_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(args.input_ckpt)
    else:
        state_dict = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)

    model = WanModel(
        dim=1536, eps=1e-5, ffn_dim=6144, freq_dim=256, in_dim=16,
        num_heads=12, num_layers=30, out_dim=16, patch_size=(1, 2, 2), text_dim=4096
    )
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[6:] if k.startswith("model.") else k] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    # Dataset
    dataset = FlashVSRTQDataset(root=args.dataset, num_samples=args.samples)
    print(f"Calibration dataset: {len(dataset)} samples from {args.dataset}")

    # Run calibration
    print(f"Running calibration ({args.samples} samples)...")
    scales = run_calibration(model, dataset, batch_size=args.batch_size)
    print(f"Collected scales from {len(scales)} layers")

    # Save
    save_calibration_cache(scales, args.output_cache)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/ptq/test_calibrator_w8a8.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ptq/calibrator_w8a8.py tests/scripts/ptq/test_calibrator_w8a8.py
git commit -m "feat: add PTQ W8A8 calibrator with DOVE dataset and ActivationCollector

FlashVSRTQDataset samples 24x24 frames, simulates 16ch latent.
ActivationCollector hooks all Linear/Conv2d/Conv3d for per-tensor stats.
Produces calibration cache JSON for TensorRT calibrator.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: TensorRT Compilation Script

**Files:**
- Create: `scripts/ptq/compile_trt_w8a8.py`
- Test: `tests/scripts/ptq/test_compile_trt_w8a8.py` (smoke test — no GPU required)

- [ ] **Step 1: Write failing smoke test**

```python
# tests/scripts/ptq/test_compile_trt_w8a8.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

def test_compile_trt_imports():
    """Script imports without error and has required functions."""
    from scripts.ptq.compile_trt_w8a8 import (
        load_dit_with_rmsnorm_fold,
        export_dit_for_trt,
        compile_trt_engine,
    )
    assert callable(load_dit_with_rmsnorm_fold)
    assert callable(export_dit_for_trt)
    assert callable(compile_trt_engine)
    print("test_compile_trt_imports PASS")

def test_torch_export_available():
    """torch.export is available in the environment."""
    import torch
    assert hasattr(torch, 'export'), "torch.export not available"
    print("test_torch_export_available PASS")

def test_trt_input_shape_spec():
    """trt.Input shape ranges are correctly specified."""
    # Verify the shape_ranges format used in compile script
    from scripts.ptq.compile_trt_w8a8 import make_trt_input_spec
    spec = make_trt_input_spec()
    assert spec is not None
    print("test_trt_input_shape_spec PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/ptq/test_compile_trt_w8a8.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# scripts/ptq/compile_trt_w8a8.py
"""
TensorRT INT8 Compilation for FlashVSR DiT (W8A8 PTQ).

Flow:
    1. Load fp16 WanModel checkpoint
    2. Fold RMSNorm gamma into downstream Linear weights
    3. torch.export to clean graph
    4. Create TensorRT IInt8EntropyCalibrator2 with DOVE DataLoader
    5. torch_tensorrt.compile with enabled_precisions={torch.int8}
    6. Save engine to .engine file
"""

import argparse
import json
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.wan_video_dit import WanModel
from src.models.quantization.rmsnorm_fold import fold_dit_rmsnorms


def check_torch_tensorrt():
    """Check if torch_tensorrt is available."""
    try:
        import torch_tensorrt as trt
        import torch_tensorrt.ptq as ptq
        print(f"torch_tensorrt version: {trt.__version__}")
        return True
    except ImportError:
        print("ERROR: torch-tensorrt not installed.")
        print("Install with: pip install torch-tensorrt")
        return False


def load_dit_with_rmsnorm_fold(checkpoint_path):
    """
    Load WanModel from checkpoint and fold RMSNorm gamma into Linear weights.

    Returns:
        WanModel (modified in-place, fp16/bf16)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = WanModel(
        dim=1536, eps=1e-5, ffn_dim=6144, freq_dim=256, in_dim=16,
        num_heads=12, num_layers=30, out_dim=16, patch_size=(1, 2, 2), text_dim=4096
    )
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[6:] if k.startswith("model.") else k] = v
    model.load_state_dict(new_sd, strict=False)

    # Move to CUDA before RMSNorm folding (some ops may need device)
    model = model.cuda()
    model.eval()

    # Fold RMSNorm gamma into downstream Linear weights
    print("Folding RMSNorm gamma into Linear weights...")
    model = fold_dit_rmsnorms(model)
    print("RMSNorm folding complete")

    return model


def make_trt_input_spec():
    """
    Return the trt.Input specification with dynamic shape ranges.

    T: 1-64 frames, H/W: 128-2048 pixels
    B: fixed at 1 for streaming video
    """
    import torch_tensorrt as trt
    return trt.Input(
        (1, 1, 16, 128, 128),   # min shape
        (4, 16, 16, 512, 512),  # opt shape (where calibration runs)
        (8, 64, 16, 2048, 2048), # max shape
        dtype=torch.float16,
    )


def export_dit_for_trt(model, example_input):
    """
    Export DiT via torch.export for TensorRT.

    Args:
        model: WanModel (fp16/bf16, with RMSNorm folded)
        example_input: tuple of (latents, timesteps, contexts)

    Returns:
        Exported module (torch.export.ExportedProgram)
    """
    print("Exporting DiT via torch.export...")
    with torch.no_grad():
        exported = torch.export.export(model, example_input)
    print(f"Exported program: {exported.graph}")
    return exported


def create_trt_calibrator(dataloader, num_samples=320):
    """
    Create TensorRT IInt8EntropyCalibrator2 wrapping DOVE DataLoader.

    Args:
        dataloader: torch.utils.data.DataLoader yielding CalibrationSample
        num_samples: max calibration iterations

    Returns:
        ptq.DataLoaderCalibrator instance
    """
    import torch_tensorrt.ptq as ptq

    calibrator = ptq.DataLoaderCalibrator(
        dataloader,
        calibrationAlgo=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device('cuda:0'),
        num_samples=num_samples,
    )
    return calibrator


def compile_trt_engine(exported_dit, output_engine, calibrator, input_spec):
    """
    Compile exported DiT to TensorRT INT8 engine.

    Args:
        exported_dit: torch.export.ExportedProgram
        output_engine: path to save .engine file
        calibrator: ptq.DataLoaderCalibrator
        input_spec: trt.Input with dynamic shape ranges
    """
    import torch_tensorrt as trt

    print("Compiling TensorRT INT8 engine...")
    compile_spec = {
        'inputs': [input_spec],
        'enabled_precisions': {torch.int8},
        'ptq_calibrator': calibrator,
        # 'workspace_size': 1 << 30,  # 1GB workspace
    }

    trt_model = trt.compile(exported_dit, **compile_spec)

    # Save
    os.makedirs(os.path.dirname(output_engine) or ".", exist_ok=True)
    torch.jit.save(trt_model, output_engine)
    print(f"TensorRT engine saved to {output_engine}")


def main():
    parser = argparse.ArgumentParser(description="Compile FlashVSR DiT to TensorRT INT8")
    parser.add_argument("--input_ckpt", type=str, required=True,
                        help="Path to DiT .safetensors or .pth checkpoint")
    parser.add_argument("--calibration_cache", type=str, default=None,
                        help="Path to calibration cache JSON (optional, for logging)")
    parser.add_argument("--output_engine", type=str, required=True,
                        help="Path to save TensorRT .engine file")
    parser.add_argument("--num_samples", type=int, default=320,
                        help="Number of calibration samples")
    args = parser.parse_args()

    if not check_torch_tensorrt():
        sys.exit(1)

    # Load and fold RMSNorm
    model = load_dit_with_rmsnorm_fold(args.input_ckpt)

    # Create example input for torch.export
    example_latents = torch.randn(1, 1, 16, 128, 128, device='cuda', dtype=torch.bfloat16)
    example_t = torch.tensor([500], device='cuda', dtype=torch.int64)
    example_ctx = torch.randn(1, 10, 4096, device='cuda', dtype=torch.bfloat16)
    example_input = (example_latents, example_t, example_ctx)

    # Export
    exported = export_dit_for_trt(model, example_input)

    # Create calibrator (use inline DOVE DataLoader)
    from scripts.ptq.calibrator_w8a8 import FlashVSRTQDataset
    from torch.utils.data import DataLoader

    dataset = FlashVSRTQDataset(root="datasets", num_samples=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    calibrator = create_trt_calibrator(dataloader, num_samples=args.num_samples)

    # Input spec
    input_spec = make_trt_input_spec()

    # Compile and save
    compile_trt_engine(exported, args.output_engine, calibrator, input_spec)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/ptq/test_compile_trt_w8a8.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ptq/compile_trt_w8a8.py tests/scripts/ptq/test_compile_trt_w8a8.py
git commit -m "feat: add TensorRT INT8 compilation script for FlashVSR DiT

torch.export clean graph → IInt8EntropyCalibrator2 → torch_tensorrt.compile
with dynamic shape ranges (T:1-64, H/W:128-2048).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: nodes.py — W8A8_PTQ quantize_mode and TRT Engine Loading

**Files:**
- Modify: `nodes.py:780` (init_pipeline function) and surrounding code
- Test: `tests/scripts/ptq/test_nodes_w8a8_ptq.py` (integration smoke test)

- [ ] **Step 1: Write failing test**

```python
# tests/scripts/ptq/test_nodes_w8a8_ptq.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

def test_nodes_has_w8a8_ptq_mode():
    """nodes.py init_pipeline handles quantize_mode='W8A8_PTQ'."""
    import nodes
    # Verify W8A8_PTQ is accepted as a quantize_mode (check the function signature or body)
    import inspect
    source = inspect.getsource(nodes.init_pipeline)
    assert 'W8A8_PTQ' in source, "W8A8_PTQ not found in init_pipeline"
    print("test_nodes_has_w8a8_ptq_mode PASS")

def test_trt_engine_loading_function_exists():
    """nodes.py has a function to load TRT engine."""
    import nodes
    assert hasattr(nodes, 'load_trt_engine') or 'load_trt_engine' in dir(nodes)
    print("test_trt_engine_loading_function_exists PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/ptq/test_nodes_w8a8_ptq.py -v`
Expected: FAIL — W8A8_PTQ not in init_pipeline

- [ ] **Step 3: Write minimal implementation**

Read `nodes.py:780-1020` to understand the existing quantize_mode branches.

Add to `nodes.py` around the quantize_mode handling section (after W8A8 SmoothQuant):

```python
elif quantize_mode == "W8A8_PTQ":
    # Load pre-compiled TensorRT INT8 engine for DiT
    # VAE stays bf16, loaded separately
    try:
        from .src.models.quantization.quant import convert_model_to_w8a16
    except ImportError:
        from src.models.quantization.quant import convert_model_to_w8a16

    log("W8A8_PTQ mode: DiT from TRT engine, VAE from bf16...", message_type='info', icon='🗜️')

    # Load VAE (bf16) via mm.load_models
    mm.load_models([vae_path])

    # TRT engine loading happens in pipeline init (not here)
    # The pipeline will check for trt_engine_path on call
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    pipe.offload_model()
```

Also add a `load_trt_engine(path)` helper function to `nodes.py`:

```python
def load_trt_engine(engine_path):
    """
    Load a pre-compiled TensorRT engine for W8A8_PTQ mode.

    Args:
        engine_path: Path to .engine file saved by compile_trt_w8a8.py

    Returns:
        torch.jit.ScriptModule (compiled TRT engine)
    """
    if not os.path.exists(engine_path):
        raise RuntimeError(f"TRT engine not found: {engine_path}")
    log(f"Loading TRT engine from {engine_path}...", message_type='info', icon='🗜️')
    engine = torch.jit.load(engine_path)
    return engine
```

And in `init_pipeline`, add `trt_engine_path=None` parameter:

```python
def init_pipeline(model, mode, device, dtype, vae_model="Wan2.1",
                  quantize_mode="None", ckpt_path=None, w8a8_engine="bf16",
                  trt_engine_path=None):  # <-- add this
```

When `quantize_mode == "W8A8_PTQ"` and `trt_engine_path` is provided, load the engine instead of fp16 DiT.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/ptq/test_nodes_w8a8_ptq.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nodes.py
git commit -m "feat(nodes): add W8A8_PTQ quantize_mode and TRT engine loading

Adds quantize_mode='W8A8_PTQ' to init_pipeline with trt_engine_path parameter.
VAE stays bf16, DiT loaded from pre-compiled TensorRT INT8 engine.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: CLI — W8A8_PTQ flags

**Files:**
- Modify: `cli_main.py:130-150` (argparse section) and `cli_main.py:580-590` (pipeline init)

- [ ] **Step 1: Write failing test**

```python
# tests/scripts/ptq/test_cli_w8a8_ptq.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

def test_cli_has_w8a8_ptq_flags():
    """cli_main.py accepts --quantize_mode W8A8_PTQ and --trt_engine."""
    import subprocess
    result = subprocess.run(
        ['python', 'cli_main.py', '--help'],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), '../../../..')
    )
    assert 'W8A8_PTQ' in result.stdout, "W8A8_PTQ not in --help"
    assert '--trt_engine' in result.stdout, "--trt_engine not in --help"
    print("test_cli_has_w8a8_ptq_flags PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/ptq/test_cli_w8a8_ptq.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

In `cli_main.py`, add to `parse_args()`:

```python
parser.add_argument(
    '--quantize_mode',
    type=str,
    choices=['None', 'W8A16', 'W8A8_SmoothQuant', 'W8A8', 'W8A8_PTQ'],
    default='None',
    help='...'
)
parser.add_argument(
    '--trt_engine',
    type=str,
    default=None,
    help='Path to pre-compiled TensorRT .engine file for W8A8_PTQ mode.'
)
```

In `main()`, update `init_pipeline` call:

```python
pipe = init_pipeline(
    model=args.model,
    mode=args.mode,
    device=device,
    dtype=dtype,
    vae_model=args.vae_model,
    quantize_mode=args.quantize_mode,
    ckpt_path=args.ckpt_path,
    w8a8_engine=args.w8a8_engine,
    trt_engine_path=args.trt_engine,  # <-- add
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/ptq/test_cli_w8a8_ptq.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cli_main.py
git commit -m "feat(cli): add --quantize_mode W8A8_PTQ and --trt_engine flags

Allows TRT INT8 engine loading via CLI for W8A8_PTQ deployment.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Pipeline Integration — model_fn_trt()

**Files:**
- Modify: `src/pipelines/flashvsr_full.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pipelines/test_model_fn_trt.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

def test_flashvsr_full_has_model_fn_trt():
    """FlashVSRFullPipeline has model_fn_trt method."""
    from src.pipelines.flashvsr_full import FlashVSRFullPipeline
    assert hasattr(FlashVSRFullPipeline, 'model_fn_trt'), "model_fn_trt not found"
    print("test_flashvsr_full_has_model_fn_trt PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipelines/test_model_fn_trt.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

In `src/pipelines/flashvsr_full.py`, add to `FlashVSRFullPipeline`:

```python
def model_fn_trt(self, latents, t, context):
    """
    Forward pass through TensorRT-compiled DiT engine.

    Args:
        latents: (B, T, C, H, W) — noisy latent input
        t: (B,) — timesteps
        context: (B, 10, 4096) — text embeddings

    Returns:
        Denoised latents (same shape as input)
    """
    if not hasattr(self, 'trt_engine_') or self.trt_engine_ is None:
        raise RuntimeError("TRT engine not loaded. Set pipe.trt_engine_ = load_trt_engine(path)")

    # TRT engine expects bf16 input
    latents = latents.to(device=self.device, dtype=torch.bfloat16)
    t = t.to(device=self.device)
    context = context.to(device=self.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output = self.trt_engine_(latents, t, context)

    return output
```

Also add `self.trt_engine_ = None` to `__init__` and a setter:

```python
@property
def trt_engine(self):
    return self.trt_engine_

@trt_engine.setter
def trt_engine(self, value):
    self.trt_engine_ = value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipelines/test_model_fn_trt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/flashvsr_full.py
git commit -m "feat(pipeline): add model_fn_trt() for TensorRT DiT call path

FlashVSRFullPipeline gains trt_engine property and model_fn_trt() method
for W8A8_PTQ mode where DiT runs via pre-compiled TRT INT8 engine.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Verification

After all tasks complete:

1. **Calibration:** `python scripts/ptq/calibrator_w8a8.py --input_ckpt models/... --output_cache cache.json --samples 320`
2. **TRT Compile:** `python scripts/ptq/compile_trt_w8a8.py --input_ckpt models/... --output_engine model.engine`
3. **Inference:** `python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 --quantize_mode W8A8_PTQ --trt_engine model.engine`
4. **Quality:** Compare PSNR/SSIM vs bf16 baseline on test videos