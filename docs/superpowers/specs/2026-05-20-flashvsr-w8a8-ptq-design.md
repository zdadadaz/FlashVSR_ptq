# PTQ W8A8 for FlashVSR — Design Spec

**Date:** 2026-05-20
**Author:** Claude
**Status:** Approved for implementation

---

## Context

FlashVSR is a video super-resolution system using a DiT (Diffusion Transformer) for latent denoising and a VAE (Video Autoencoder) for encoding/decoding. The goal is full INT8 quantization (W8A8) for all compute-heavy ops in the DiT, with the VAE remaining in bf16/fp16. Deployment target is TensorRT INT8 via `torch_tensorrt.ptq`.

**Why hybrid (DiT INT8 + VAE bf16):** The DiT carries ~95% of compute in video diffusion models. The VAE is relatively lightweight and highly sensitive to quantization noise — keeping it in bf16 preserves output quality.

---

## Quantization Scheme

| Component | Weight | Activation |
|-----------|--------|------------|
| Weight | **Symmetric** int8 — `scale = absmax(w) / 127`, `zero_point = 0` | N/A |
| Activation | N/A | **Asymmetric** int8 per-tensor — `scale = (max - min) / 255`, `zero_point = round(-min / scale)` |

This differs from the previous SmoothQuant approach which migrated activation difficulty to weights via a heuristic alpha. Here we use direct asymmetric activation quantization (standard PTQ).

---

## Architecture

### Two-Part Model Strategy

```
Input Video
    │
    ▼
┌─────────────────────────────────┐
│  VAE (bf16/fp16, unchanged)     │
│  • encode: RGB → latent         │
│  • decode: latent → RGB         │
└─────────────────────────────────┘
    │ latent
    ▼
┌─────────────────────────────────┐
│  DiT (W8A8 via TensorRT INT8)   │
│  • denoise latent via WanModel  │
│  • 30 DiTBlocks × 30 layers     │
│  • weight: symmetric int8        │
│  • activation: asymmetric int8    │
└─────────────────────────────────┘
    │
    ▼
Output Video
```

### DiT Quantized Layer Strategy

Every `nn.Linear`, `nn.Conv2d`, `nn.Conv3d` in the DiT gets quantized. Non-matmul ops handled as follows:

**RMSNorm (in DiTBlock):**
- RMSNorm has learnable `weight` (γ gain) and optional `bias` — both are static (no input dependence)
- These can be **folded** into the downstream Linear's weight/bias
- The dynamic RMS computation `x_rms = sqrt(mean(x²) + eps)` must remain as a runtime op — TensorRT auto-fuses it with the downstream Linear into a single CUDA kernel
- Formula: `output = x * (γ / rms(x)) + bias` where `rms(x) = sqrt(mean(x²) + eps)`
- After folding γ into Linear: new Linear weight = `original_weight * (γ / rms_val_at_calibration_time)` — but since rms_val is dynamic, we only fold γ statically

**Foldable:** `γ` (gain), `bias` — multiply into Linear weight/bias before quantization
**Not foldable:** `x_rms` — pure runtime computation, TensorRT fuses automatically

**GELU / SiLU / activation functions:** No quantization needed. These are element-wise ops that TensorRT handles in bf16 within the fused kernel. They do not appear as separate quantized ops.

**Residual additions (+x):** The residual addition is a pure bf16 op — TensorRT handles it as part of the fused compute graph without separate quantization.

---

## Calibration Pipeline

### Dataset

- **Source:** DOVE dataset at `datasets/DOVE/train/HQ-VSR`
- **Samples:** 320 frames (recommended per PyTorch TensorRT docs)
- **Preprocessing:** Sample 24×24 frames, convert to latent-simulated input (16ch, 24×24 spatial)
- **Timestep distribution:** Mix multiple timesteps (e.g., 0, 200, 500, 800, 999) to cover denoising range
- **Batch size:** 32

### Calibration DataLoader

```python
@dataclass
class CalibrationSample:
    latents: torch.Tensor      # (B, T, 16, 24, 24), bf16
    timesteps: torch.Tensor    # (B,), int64
    contexts: torch.Tensor     # (B, 10, 4096), bf16

class FlashVSRTQDataset(torch.utils.data.Dataset):
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> CalibrationSample: ...
```

### Calibration Flow

```
1. Load fp16 DiT from safetensors checkpoint
   │
2. Fold RMSNorm's learnable γ and bias into adjacent Linear weights/biases
   │
3. Run calibration DataLoader through DiT (forward only, no gradient)
   │
4. Collect per-layer activation min/max (per-tensor over full activation tensor)
   │
5. Compute act_scale = (max - min) / 255, zero_point = round(-min / act_scale)
   │
6. Quantize DiT weights to symmetric int8 using absmax / 127
   │
7. Save calibration cache (JSON): layer_name → {act_scale, zero_point, weight_scale}
```

**Note:** No custom QDWrapper inserted into the PyTorch graph. TensorRT's `IInt8EntropyCalibrator2` handles Q/DQ insertion during compilation.

---

## TensorRT Compilation

### Flow

```
1. Take calibrated DiT (weights int8, scales stored in buffers)
   │
2. Export DiT via torch.export (PyTorch 2.0) or ONNX
   → NOT torch.jit.trace (breaks on dynamic scale buffers)
   │
3. Create TensorRT IInt8EntropyCalibrator2 wrapping the DOVE DataLoader
   │
4. torch_tensorrt.compile(
       model=exported_dit,
       inputs=[trt.Input((B, T, C, H, W))],
       enabled_precisions={torch.int8},
       ptq_calibrator=calibrator,
   )
   │
5. Save TensorRT engine (.engine or .pt)
```

### VAE Handling

VAE is NOT compiled into the TensorRT engine. It remains as a separate PyTorch bf16 module executed outside the TRT engine. The pipeline coordinates execution:
1. VAE.encode → latent (bf16)
2. TensorRT DiT engine → denoised latent (int8 internal, bf16 output)
3. VAE.decode → RGB frames (bf16)

---

## Component Inventory

### New Files

| File | Purpose |
|------|---------|
| `src/models/quantization/ptq_w8a8.py` | W8A8 quantized layer classes (SymmetricWeightLinear, AsymmetricActLinear, QuantizedConv3d) |
| `src/models/quantization/rmsnorm_fold.py` | Fold RMSNorm γ/bias into Linear weights |
| `scripts/ptq/calibrator_w8a8.py` | CalibrationDataLoader, ActivationCollector, calibration run |
| `scripts/ptq/convert_w8a8.py` | Load checkpoint, fold RMSNorm, quantize weights, save cache |
| `scripts/ptq/compile_trt_w8a8.py` | torch.export → TensorRT compile with calibrator |

### Modified Files

| File | Change |
|------|--------|
| `nodes.py` | Add `quantize_mode="W8A8_PTQ"` option that loads a pre-calibrated TRT engine instead of running fp16 DiT |
| `src/pipelines/flashvsr_full.py` | Add `model_fn_trt()` for TensorRT DiT call path |
| `cli_main.py` | Add `--quantize_mode W8A8_PTQ --trt_engine PATH` flags |

---

## Quantized Layer Classes

### SymmetricWeightLinear (W8A16 — fallback)

Weight: int8 symmetric, activation: bf16 passthrough. Identical to existing `SymmetricWeightLinear` in `ptq.py`.

### AsymmetricActLinear (W8A8)

Weight: int8 symmetric, activation: int8 asymmetric per-tensor.
```python
# Scale: act_scale = (max - min) / 255, zero_point = round(-min / act_scale)
x_int8 = round((x - zero_point) / act_scale)
y = F.linear(x_int8.float(), weight_int8 * weight_scale, bias)
```

### QuantizedConv3d (W8A8)

Same pattern as AsymmetricActLinear but for Conv3d:
```python
x_int8 = round((x - zero_point) / act_scale)
y = F.conv3d(x_int8, weight_int8 * weight_scale, bias, ...)
```

---

## RMSNorm Fold Utility

```python
def fold_rmsnorm_into_linear(rmsnorm: RMSNorm, linear: nn.Linear) -> nn.Linear:
    """
    Fold RMSNorm's learnable gamma (weight) and bias into the adjacent Linear.
    The dynamic RMS(x) computation is NOT folded — it remains as a runtime op.
    TensorRT will auto-fuse RMSNorm(pure) + Linear(INT8) into a single kernel.
    """
    gamma = rmsnorm.weight.data  # (C,)
    bias = rmsnorm.bias.data if rmsnorm.bias is not None else None

    # Fold gamma into linear weight: w_folded = w * gamma.view(1, -1, 1, 1) / rms_val
    # But we can only fold gamma statically: w_folded = w * gamma.view(...)
    # rms_val is dynamic, so fold gamma only (not rms)
    w_folded = linear.weight.data * gamma.view(1, -1, 1, 1)

    new_linear = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    new_linear.weight.data = w_folded
    if bias is not None and linear.bias is not None:
        new_linear.bias.data = linear.bias.data + bias
    elif bias is not None:
        new_linear.bias = nn.Parameter(bias.clone())

    return new_linear
```

---

## Verification Plan

1. **Calibration:** `python scripts/ptq/calibrator_w8a8.py --input_ckpt model.safetensors --output_cache cache.json --samples 320`
2. **Conversion:** `python scripts/ptq/convert_w8a8.py --input_ckpt model.safetensors --calibration_cache cache.json --output_ptq model_w8a8.pt`
3. **TRT Compile:** `python scripts/ptq/compile_trt_w8a8.py --input_ptq model_w8a8.pt --output_engine model.trt`
4. **Inference:** `python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 --quantize_mode W8A8_PTQ --trt_engine model.trt`
5. **Quality check:** Compare PSNR/SSIM vs bf16 baseline on test videos

---

## Dependencies

- `torch>=2.0`
- `torch-tensorrt` (`pip install torch-tensorrt`)
- `torchvision` (for DataLoader utilities)
- DOVE dataset at `datasets/DOVE/`

If `torch-tensorrt` is unavailable, fall back to native PyTorch quantization (`torch.quantization.quantize_dynamic`) with the same W8A8 layer classes — quality may degrade but still functional.