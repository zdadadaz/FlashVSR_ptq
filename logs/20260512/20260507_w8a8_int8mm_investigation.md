# W8A8 Int8MatmulLinear Investigation — 2026-05-07

## Finding: True W8A8 vs W8A16 (Weight-Only)

### Problem
Tested `Int8MatmulLinear` using `torch._int_mm` for actual INT8 matmul. Got 13 dB PSNR vs FP16 — catastrophic quality loss.

### Root Cause Discovered

**Original `Int8ActLinear` is NOT true W8A8** — it's actually W8A16 (weight-only int8) with bf16 activations:

```python
# Int8ActLinear.forward() — actual behavior:
x_bf16 = x.to(w_dtype)           # NO activation quantization! just dtype cast
w_bf16 = self.weight * self.weight_scale  # only weight dequantized to bf16
y = F.linear(x_bf16, w_bf16, bias)  # bf16 matmul — NO int8 tensor cores used
```

The "W8A8" name was misleading — it stored int8 weights and had `act_scale` buffers, but **never quantized activations to int8**.

### True W8A8 Path (Int8MatmulLinear)

```python
# Int8MatmulLinear.forward() — true W8A8:
x_int8 = torch.round(x / self.act_scale).to(torch.int8)  # ACTUAL activation quantization
out_int32 = torch._int_mm(x_flat, self.weight.t())        # INT8 matmul via tensor cores
out = out_int32 * self.weight_scale                       # rescale
```

### Results Comparison

| Class | Activation Quantization | Matmul Type | FPS (scale 4) | PSNR vs FP16 |
|-------|------------------------|-------------|---------------|--------------|
| `Int8ActLinear` | ❌ NO (bf16 pass-through) | bf16 | 5.71 | **36.96 dB** ✅ |
| `Int8MatmulLinear` | ✅ YES (int8 quantized) | INT8 (_int_mm) | 6.81 | **13.43 dB** ❌ |

### Why True W8A8 Fails on FlashVSR

1. **Calibration mismatch**: `act_amax` collected during calibration (4 frames per video, simple inference) doesn't represent full activation range during diffusion denoising (31 frames, iterative noise removal)

2. **Static scale problem**: Activation scales are fixed after calibration but activations during actual inference vary significantly across diffusion timesteps

3. **Error accumulation**: Quantizing activations to int8 loses precision that accumulates through 30 DiT layers

### Conclusion

- **W8A16 (Int8ActLinear / WeightOnlyInt8Linear)**: Works well for FlashVSR (35-37 dB PSNR)
- **True W8A8 (Int8MatmulLinear)**: Requires better calibration (per-timestep activation stats) or SmoothQuant to work properly
- The name "W8A8" in original Int8ActLinear was misleading — it was really "W8A16"

### Recommendation

For production: Use `W8A16` (weight-only int8) which achieves 35+ dB PSNR with memory savings. True W8A8 requires either:
1. Per-timestep activation calibration
2. SmoothQuant migration (but this failed earlier with only 9 dB)
3. Per-head QKV split with independent scales

## Files Modified
- `src/models/quantization/quant.py`: Added `Int8MatmulLinear` class with `torch._int_mm`
- Reverted `convert_model_to_w8a8()` to use `Int8ActLinear` instead of `Int8MatmulLinear`

## Log Files
- `logs/w8a8_int8mm_final_114214.log` - Full W8A8 int8mm test run
- `logs/w8a8_int8mm_test2_111709.log` - Earlier test run