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
| Activation | N/A | **Asymmetric** int8 per-tensor (default) — `scale = (max - min) / 255`, `zero_point = round(-min / scale)` |

**Activation note:** For Ampere/RTX 30-50 series, asymmetric (`zero_point ≠ 0`) has a slight Tensor Core throughput penalty vs symmetric. Default to asymmetric for maximum quality; if PSNR gain is < 0.05dB over symmetric, switch to symmetric (`zero_point = 0`) for full Tensor Core throughput.

This differs from the previous SmoothQuant approach which migrated activation difficulty to weights via a heuristic alpha. Here we use direct asymmetric activation quantization (standard PTQ) with TensorRT handling Q/DQ insertion automatically.

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

**Note:** The model stays in pure fp16/bf16 — no custom quantized layer classes inserted in PyTorch. TensorRT's `IInt8EntropyCalibrator2` handles Q/DQ insertion during compilation (Step 4 only collects stats). RMSNorm fold is the only graph modification before export.

---

## TensorRT Compilation

### Flow

```
1. Load fp16 DiT from safetensors checkpoint (pure fp16/bf16, no quantization in PyTorch)
   │
2. Fold RMSNorm's learnable γ and bias into adjacent Linear weights/biases
   │
3. Export DiT via torch.export (PyTorch 2.0) or ONNX
   → NOT torch.jit.trace (breaks on dynamic scale buffers)
   │
4. Create TensorRT IInt8EntropyCalibrator2 wrapping the DOVE DataLoader
   │
5. torch_tensorrt.compile(
       model=exported_dit,
       inputs=[trt.Input((B, T, 16, H, W), dtype=trt.float16,
              shape_ranges=[((1, 1, 16, 128, 128), (4, 16, 16, 512, 512), (8, 64, 16, 2048, 2048)])],
       # T: 1-64, H/W: 128-2048 — dynamic dims to handle variable video resolutions
       enabled_precisions={torch.int8},
       ptq_calibrator=calibrator,
   )
   │
6. Save TensorRT engine (.engine)
```

**No pre-quantization in PyTorch.** The DiT remains fp16/bf16 throughout — weights stay float. TensorRT's calibrator collects activation statistics during compilation, then inserts Q/DQ nodes and quantizes weights internally. This ensures TensorRT sees a clean graph and can apply optimal fusion.

**Dynamic shapes:** T (frames), H/W (resolution) are declared as dynamic ranges in `trt.Input` so a single compiled engine handles variable video sizes without recompilation. B (batch) is typically fixed at 1 for streaming video.

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
| `src/models/quantization/rmsnorm_fold.py` | Fold RMSNorm γ/bias into Linear/Conv weights |
| `scripts/ptq/calibrator_w8a8.py` | CalibrationDataLoader, ActivationCollector, calibration run |
| `scripts/ptq/compile_trt_w8a8.py` | torch.export → TensorRT compile with calibrator |

**Note:** No custom W8A8 layer classes (e.g. `AsymmetricActLinear`) are inserted into the PyTorch graph. Quantization happens entirely inside TensorRT during compilation. The existing `src/models/quantization/ptq.py` (`SymmetricWeightLinear`, `AsymmetricActLinear`) is kept as a standalone reference for native-PyTorch fallback only.

### Modified Files

| File | Change |
|------|--------|
| `nodes.py` | Add `quantize_mode="W8A8_PTQ"` option that loads a pre-compiled TRT engine instead of running fp16 DiT |
| `src/pipelines/flashvsr_full.py` | Add `model_fn_trt()` for TensorRT DiT call path |
| `cli_main.py` | Add `--quantize_mode W8A8_PTQ --trt_engine PATH` flags |

---

## RMSNorm Fold Utility

```python
def fold_rmsnorm_into_linear(rmsnorm, layer):
    """
    Fold RMSNorm's learnable gamma (weight) and bias into the adjacent Linear or Conv.

    Foldable: gamma (gain) and bias — static, no input dependence
    NOT foldable: dynamic RMS(x) computation — must remain as runtime op.
                   TensorRT auto-fuses RMSNorm(pure) + Linear into a single kernel.

    Important: WanModel's RMSNorm typically has elementwise_affine=True but bias=None
    (standard DiT practice). If bias is present and non-zero, it must be propagated
    through the layer's weight (e.g. for Linear: bias @ weight.T). Since this is rare
    in modern DiTs, we assert bias is None or all-zeros rather than implementing the
    full matrix multiply. If non-zero bias is encountered, only fold gamma.

    Linear weight shape: (out_features, in_features)
    → gamma shapes the in_features axis → broadcast over dim=1 (columns)
    """
    gamma = rmsnorm.weight.data
    bias = rmsnorm.bias.data if rmsnorm.bias is not None else None

    # If bias is non-trivial, only fold gamma (skip bias folding to avoid
    # incorrect results without implementing full weight @ bias.T reshape)
    if bias is not None and not (bias.abs() < 1e-6).all():
        import warnings
        warnings.warn(f"RMSNorm bias is non-zero ({bias.abs().max():.6f}), folding gamma only. "
                      "Bias folding requires weight @ bias.T — implement if needed.")
        bias = None

    if isinstance(layer, nn.Linear):
        # gamma acts on in_features (columns of weight matrix)
        w_folded = layer.weight.data * gamma.view(1, -1)   # (out, in) * (1, in) → (out, in)
        new_layer = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
        new_layer.weight.data = w_folded
        if bias is not None and layer.bias is not None:
            new_layer.bias.data = layer.bias.data + bias
        elif bias is not None:
            new_layer.bias = nn.Parameter(bias.clone())

    elif isinstance(layer, nn.Conv2d):
        # gamma acts on in_channels (dim=1 of weight [out_c, in_c, kH, kW])
        w_folded = layer.weight.data * gamma.view(1, -1, 1, 1)
        new_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                              stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                              groups=layer.groups, bias=layer.bias is not None,
                              padding_mode=layer.padding_mode)
        new_layer.weight.data = w_folded
        if bias is not None and layer.bias is not None:
            new_layer.bias.data = layer.bias.data + bias
        elif bias is not None:
            new_layer.bias = nn.Parameter(bias.clone())

    elif isinstance(layer, nn.Conv3d):
        # gamma acts on in_channels (dim=1 of weight [out_c, in_c, kT, kH, kW])
        w_folded = layer.weight.data * gamma.view(1, -1, 1, 1, 1)
        new_layer = nn.Conv3d(layer.in_channels, layer.out_channels, layer.kernel_size,
                              stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                              groups=layer.groups, bias=layer.bias is not None)
        new_layer.weight.data = w_folded
        if bias is not None and layer.bias is not None:
            new_layer.bias.data = layer.bias.data + bias
        elif bias is not None:
            new_layer.bias = nn.Parameter(bias.clone())

    return new_layer
```

---

## Verification Plan

1. **Calibration:** `python scripts/ptq/calibrator_w8a8.py --input_ckpt model.safetensors --output_cache cache.json --samples 320`
2. **TRT Compile:** `python scripts/ptq/compile_trt_w8a8.py --input_ckpt model.safetensors --calibration_cache cache.json --output_engine model.trt`
   - This single step: loads fp16 DiT → folds RMSNorm → runs torch.export → compiles with TensorRT calibrator → saves .engine
3. **Inference:** `python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 --quantize_mode W8A8_PTQ --trt_engine model.trt`
4. **Quality check:** Compare PSNR/SSIM vs bf16 baseline on test videos

---

## Dependencies

- `torch>=2.0`
- `torch-tensorrt` (`pip install torch-tensorrt`)
- `torchvision` (for DataLoader utilities)
- DOVE dataset at `datasets/DOVE/`

If `torch-tensorrt` is unavailable, fall back to native PyTorch quantization (`torch.quantization.quantize_dynamic`) with the same W8A8 layer classes — quality may degrade but still functional.