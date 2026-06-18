# Plan: REDS-based PTQ a8w8 + Output QDQ Rebenchmark

**Date**: 2026-06-12 23:30
**Goal**: Use full REDS train_sharp (240 sequences × 100 frames @ 1280×720) to generate LQ via OpenCV bicubic ×4 downsample, then re-run PTQ a8w8 calibration + conversion + PSNR vs FP16, on top of the output-QDQ contract from previous round.

## Why the previous benchmark was bad (recap)
- 5 small videos × 8 samples → ~40 calibration samples total
- Per-channel output scale (1536 floats per layer) cannot be calibrated from 8 samples — inter-timestep dynamic range is huge
- Static per-channel output scale clips ~32% of channels (max 25× over bound)
- Round 5 (multiplier 5×) only reached 16.74 dB

## Why REDS is the right calibration source
- 240 sequences × 100 frames = 24,000 frames of **real video content** (vs. synthetic 5 videos)
- Per-sequence content diversity ensures activation distribution coverage
- We control the LQ generation (bicubic ×4) so it matches inference-time LQ distribution exactly
- Latent via VAE encoding is still the correct contract (preserves per-channel distribution)

## Step 1: Generate LQ videos from REDS
- Script: `scripts/ptq/build_reds_lq_videos.py` (new)
- For each sequence in `/home/user/data/REDs/train/train_sharp/*/`, load 100 PNGs at 1280×720
- Apply `cv2.resize(frame, (W//4, H//4), interpolation=cv2.INTER_CUBIC)` → 320×180 LQ
- Encode as mp4 (libx264, crf=20) at native fps or 30 fps
- Output: `/home/user/data/REDs/train/LQ/{seq_name}.mp4`
- Total: 240 mp4 files, ~800s of video

## Step 2: Extend calibrator to accept video dataset path
- `scripts/ptq/fakequant_calibrate.py` already has `--dataset_train` that recursively glob `*.mp4`
- Just point it to `/home/user/data/REDs/train/LQ/` with `--num_videos 240`
- Use `--num_samples 256` (covers 1+ sequence worth of forward passes)
- Use `--calib_frames 16` (16 frames per video, more efficient than 32 with 240 videos)
- VAE encode still required for accurate latent distribution
- Output: `outputs/2026-06-13_reds_ptq_a8w8/calib_cache.json` (schema v2 with output_stats)

## Step 3: Convert with mode 7 (static_tensor_symmetric)
- Use existing `scripts/ptq/fakequant_convert.py` 
- `--quantize_mode FakeQuant_A8W8_DRAQ` for backward-compat OR add new mode alias
- Key CLI flags from previous round:
  - `--output_scale_multiplier 1.0` (we'll set the calibration correctly this time)
  - `--enable_smoothquant` (off by default per requirements)

## Step 4: Run FP16 teacher + PTQ a8w8 inference on 5 held-out REDS sequences
- Select 5 LQ videos not in calibration (or use them; calibration is offline) — use first 5 sequences for eval
- Run `cli_main.py --input {LQ} --output {fp16} --scale 4 --mode tiny --vae_model Wan2.1 ...`
- Then run with `--quantize_mode FakeQuant_A8W8_DRAQ` for PTQ

## Step 5: PSNR vs FP16
- `scripts/compare_video_psnr.py` (already exists)
- Compare per-frame PSNR, average

## Expected outcomes
- With 256 samples spread over 240 diverse sequences, the per-channel output scale should be much more accurate
- Target PSNR: ≥ 26 dB (close to PTQ project's 30 dB)
- Even with 16 frames per video, 256 samples × diverse content > previous 40 samples × 5 same-source videos

## Risks
- Calibrator with 256 samples × 16 frames × 240 videos could be slow (~30 min on RTX 4090)
  - Mitigate: per video encode is dominant cost; calibrate only 16 frames per video
- VAE encode is also expensive (one forward per frame, 16 frames × 240 videos = 3840 forward passes)
  - Mitigate: cache LQ→latent on disk OR calibrate with fewer videos
- Output QDQ on top of static per-channel may still clip — **use per-tensor scalar** for output stats (collapse helper from previous round)

## Architecture
- No code changes to the FakeQuant model contract (we already added mode 7, per-tensor collapse helpers)
- Only **new code**: the REDS→LQ video builder script
- Calibrator CLI: re-use existing flags
