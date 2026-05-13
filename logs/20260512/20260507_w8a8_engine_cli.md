# W8A8 Engine CLI Argument — 2026-05-07

## Summary

Added `--w8a8_engine` CLI argument to allow users to choose between W8A8 inference engines:

- `bf16` (default): Uses `Int8ActLinear` with bf16 matmul (~37 dB PSNR, better quality)
- `int8mm`: Uses `Int8MatmulLinear` with `torch._int_mm` (~13 dB PSNR, faster but experimental)

## Changes

### CLI (`cli_main.py`)
- Added `--w8a8_engine` argument with choices `['bf16', 'int8mm']`
- Passes `w8a8_engine` to `init_pipeline()`

### Nodes (`nodes.py`)
- Updated `init_pipeline()` signature to accept `w8a8_engine` parameter
- Log message now shows engine being used

### Quantization (`src/models/quantization/quant.py`)
- Updated `convert_model_to_w8a8()` signature: `engine='bf16'`
- Function body now uses conditional:
  - `engine='int8mm'`: creates `Int8MatmulLinear`
  - `engine='bf16'`: creates `Int8ActLinear` (default)

## Usage

```bash
# Default W8A8 (bf16 engine)
python cli_main.py --input video.mkv --output upscaled.mp4 --quantize_mode W8A8

# Explicit bf16 engine
python cli_main.py --input video.mkv --output upscaled.mp4 --quantize_mode W8A8 --w8a8_engine bf16

# int8mm engine (experimental, lower quality)
python cli_main.py --input video.mkv --output upscaled.mp4 --quantize_mode W8A8 --w8a8_engine int8mm
```

## Files Modified
- `cli_main.py`
- `nodes.py`
- `src/models/quantization/quant.py`
- `README.md`