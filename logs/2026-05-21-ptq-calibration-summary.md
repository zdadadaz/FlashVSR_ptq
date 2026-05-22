# FlashVSR PTQ W8A8 Calibration - Summary

**Date:** 2026-05-21
**Author:** Claude Code

## What Was Done

### 1. Fixed `calibrator_w8a8.py` - Multiple Issues

**Problem:** Calibration script failed with various errors.

**Fixes Applied:**

1. **Added `sys.path` for imports** (line 7-13)
   - Added project root to Python path so `src.models.wan_video_dit` imports work

2. **Fixed frame_size to 64x64** (line 47)
   - Original 24x24 failed window partitioning assertion (requires H,W % 16 == 0)
   - Window partitioning uses win=(2,8,8), after patchify stride (1,2,2): 64→32, 32%8=0 ✓

3. **Fixed num_frames to 6** (line 48)
   - Original 1 frame failed `is_stream` assertion (requires f=6 when pre_cache is None)
   - 6 is divisible by 2 (win[0]=2) ✓

4. **Fixed latents shape handling** (line 225-228)
   - Dataset returns (T, C, H, W), model expects (B, C, T, H, W)
   - Permute + unsqueeze to convert: `x.permute(1, 0, 2, 3).unsqueeze(0)`

5. **Added CUDA support** (line 311-317)
   - Model moved to CUDA for calibration
   - Cross-attention KV cache initialization with dummy context

6. **Rewrote calibration loop to use `model_fn_wan_video` pattern** (line 240-280)
   - Direct `model.forward()` has a bug where `SelfAttention` returns 1 value when `is_stream=False`
   - Pipeline's `model_fn_wan_video` handles this correctly by calling blocks individually
   - Precomputes timestep embedding, RoPE frequencies manually
   - Uses try/except to handle both is_stream=True and is_stream=False return patterns

7. **Added `sinusoidal_embedding_1d` import** (line 27)
   - Was missing at top level, only imported inside `main()`

### 2. Created venv for execution
- `python3.10 -m virtualenv --system-site-packages .venv`
- Uses system site packages (torch, numpy, etc.)

### 3. Calibration completed successfully
- 320 samples processed
- 244 layers calibrated
- Cache saved to `./calibration_cache.json` (1465 lines)

## Remaining Steps

### Step 2: TRT Compilation
- **Blocked:** torch-tensorrt installation corrupted venv
- **Solution needed:** User deleted venv, need to recreate
- After recreation, run:
  ```bash
  python scripts/ptq/compile_trt_w8a8.py \
    --input_ckpt models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
    --output_engine /tmp/model.engine \
    --calibration_cache ./calibration_cache.json
  ```

### Step 3: Inference Test
- After TRT engine is built, run:
  ```bash
  python cli_main.py \
    --input data/lowres/city2_1.mp4 \
    --output ./outputs/city2_1_upscaledx4.mp4 \
    --scale 4 \
    --quantize_mode W8A8_PTQ \
    --trt_engine /tmp/model.engine \
    --mode tiny
  ```

## Known Issues in Model Code

**`SelfAttention.forward` return value bug (line 448-450):**
```python
if is_stream:
    return self.o(x), cache_k, cache_v
return self.o(x)  # Should return 3 values, but returns 1
```

The model expects 3 values from `self.self_attn()` in `DiTBlock.forward` (line 532), but `SelfAttention.forward` only returns 3 values when `is_stream=True`. This is a bug in the model code that required the try/except workaround in the calibrator.

## Files Modified
- `scripts/ptq/calibrator_w8a8.py` - Multiple fixes for calibration to work

## Files Created
- `.venv/` - Python virtual environment
- `calibration_cache.json` - PTQ calibration cache (244 layers, 1465 lines)