# PR2 Static Mixed PTQ Conversion/Eval Sweep

Date: 2026-06-18
Branch: feat/static-mixed-qparams-pr2

## Scope

Completed the second PR from `20260618_171210_flashvsr_static_mixed_a8w8_a16w8_qat_plan`:

- Added `scripts/experiments/static_mixed_qat/run_policy_sweep.py` to generate top-percent A16 fallback policies, convert each policy to inference-format FakeQuant checkpoints, write manifests, and update the leaderboard.
- Fixed leaderboard handling so rows are upserted by `run_id` and `compare_video_psnr.py` JSON (`psnr_avg_db`) is parsed.
- Ran mixed static top10/top20/top40/top60 policy conversion using existing REDS static tensor symmetric calibration and true-sensitivity ranking.
- Ran same-run FP16 vs static-mixed inference evaluation on `data/lowres/bowing_cif.mp4` first 16 frames.

## Inputs

- FP checkpoint: `models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors`
- Calibration cache: `outputs/2026-06-12_a8w8_tensor_sym/calib_cache.json`
- Sensitivity cache: `outputs/true_sensitivity/sensitivity_per_layer.json`
- Activation QDQ mode: `static_tensor_symmetric`
- Clipping: `minmax`
- Scope: DiT Linear only; Wan VAE remains unquantized.

## Conversion artifacts

Run directory: `outputs/static_mixed_qat/20260618_pr2_policy_sweep`

| Policy | A8W8 layers | A16W8 layers | Checkpoint |
|---|---:|---:|---|
| top10 A16 | 275 | 31 | `checkpoints/static_mixed_top10_minmax.safetensors` |
| top20 A16 | 244 | 62 | `checkpoints/static_mixed_top20_minmax.safetensors` |
| top40 A16 | 183 | 123 | `checkpoints/static_mixed_top40_minmax.safetensors` |
| top60 A16 | 122 | 184 | `checkpoints/static_mixed_top60_minmax.safetensors` |

Every conversion summary reported `converted=306`, `fallback=0`, `fp16_skip=0`.

## Inference/evaluation

Reproduction script:

```bash
bash outputs/static_mixed_qat/20260618_pr2_policy_sweep/run_inference_eval.sh
```

Eval setup:

- Input: `data/lowres/bowing_cif.mp4`
- Frames: first 16
- Mode: `tiny`
- Scale: 4x
- VAE: `Wan2.1`
- Precision: bf16
- Comparison: same-run FP16 output vs quantized output via `scripts/compare_video_psnr.py ref dist --out-json`.

PSNR vs FP16:

| Policy | PSNR avg dB |
|---|---:|
| top10 A16 | 13.7097505585 |
| top20 A16 | 13.5358511713 |
| top40 A16 | 13.5122019740 |
| top60 A16 | 15.5816900220 |

## Validation

```bash
.venv/bin/python -m pytest \
  tests/scripts/experiments/test_static_mixed_leaderboard.py \
  tests/scripts/experiments/test_static_mixed_policy_sweep.py \
  tests/scripts/ptq/test_static_mixed_policy_generator.py -q
# 8 passed in 0.62s

.venv/bin/python -m pytest \
  tests/test_output_qdq.py \
  tests/test_qat_pipeline.py \
  tests/test_fakequant_august_recovery.py \
  tests/test_static_ptq_baseline.py \
  tests/test_lsgquant_draq_static.py \
  tests/scripts/ptq/test_lsgquant_volts_policy.py -q
# 60 passed in 3.92s
```

## Conclusion

The NPU-compatible static mixed conversion/eval path works end-to-end, but minmax `static_tensor_symmetric` with sensitivity-derived top10/top20/top40 A16 fallback remains poor (~13.5-13.7 dB). Top60 A16 improves to ~15.58 dB, but is still far from dynamic/DRAQ quality. PR3 should prioritize static-specific clipping and correction: percentile/MSE clipping, plus opt-in bias correction, and ideally static-specific layer sensitivity rather than only dynamic-asymmetric sensitivity.
