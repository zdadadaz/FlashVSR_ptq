# Bug Fix: UnboundLocalError in flashvsr - 2026-05-13

## Issue
Command was failing with:
```
UnboundLocalError: local variable 'final_output_tensor' referenced before assignment
```

## Root Cause
In `flashvsr()` function (`nodes.py` line 1501), when `chunk_size > 0` but `chunk_size >= total_frames`, the code takes the `else` branch (lines 1455-1491). The OOM recovery loop in this branch did NOT initialize `final_output_tensor = None` before entering the loop, unlike the chunked path which initializes `final_outputs = []` before the loop.

When OOM occurred on the first attempt and all optimizations were exhausted (tiled_vae=True, tiled_dit=True, unload_dit=True), the exception was re-raised but `final_output_tensor` was never assigned because the loop only had one iteration before raising.

## Fix Applied
Added `final_output_tensor = None` initialization before the while loop in the non-chunked path (line 1461).

Also improved the error handling to:
1. Log the actual optimization state when all retries are exhausted
2. Raise a clear RuntimeError instead of letting the error propagate silently

## Changes
File: `nodes.py`
- Line ~1457-1461: Added `final_output_tensor = None` before the retry loop
- Lines ~1486-1491: Enhanced error message to include optimization state

## Verification
After fix, when OOM occurs with all optimizations enabled, the error message is:
```
RuntimeError: Processing failed: unable to generate output due to insufficient VRAM even with all optimizations enabled
```

## Secondary Finding
Even after fixing the UnboundLocalError, the process still hits OOM because:
1. VRAM estimation (5.4GB) is inaccurate - actual usage is much higher
2. `full` mode with W8A16 and small frame chunks (4 frames) exceeds VRAM capacity
3. The 23.48GB GPU gets overwhelmed by the W8A16 model + full pipeline

## Working Configuration
For the user's GPU (23.48GB), use `tiny-long` mode with aggressive optimization:
```bash
python3.10 cli_main.py \
  --input data/lowres/animal_2.mp4 \
  --output animal_2_upscaledx4_tiny_long_w8a16.mp4 \
  --scale 4 \
  --mode tiny-long \
  --quantize_mode W8A16 \
  --ckpt_path models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors \
  --no_color_fix \
  --frame_chunk_size 4 \
  --tiled_vae \
  --resize_factor 0.5
```