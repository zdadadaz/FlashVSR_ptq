# PR3 Static Clipping + Bias Correction

Date: 2026-06-18
Branch: feat/static-mixed-qparams-pr3

## Scope

Implemented PR3 from the static mixed A8W8/A16W8 QAT/PTQ plan:

- Added static activation clipping policy support to `scripts/ptq/fakequant_convert.py` for `static_tensor_symmetric`:
  - `minmax`
  - `percentile_99`
  - `percentile_999`
  - `mse` cache-only candidate selection over minmax/p99/p99.9
- Wired `--static_clipping` into the conversion CLI and static mixed sweep runner.
- Kept bias correction as explicit opt-in via `--enable_bias_correction` and verified it is recorded in conversion summaries/leaderboard rows.
- Added CPU contract tests for clipping math and command construction.

## Experiment

Run directory: `outputs/static_mixed_qat/20260618_pr3_clipping_bias_sweep`

Reproduce conversion:

```bash
.venv/bin/python -u scripts/experiments/static_mixed_qat/run_policy_sweep.py \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  --calibration_cache outputs/2026-06-12_a8w8_tensor_sym/calib_cache.json \
  --sensitivity_json outputs/true_sensitivity/sensitivity_per_layer.json \
  --out_dir outputs/static_mixed_qat/20260618_pr3_clipping_bias_sweep \
  --a16_percent 60 \
  --activation_qdq_mode static_tensor_symmetric \
  --clipping mse \
  --enable_bias_correction \
  --eval_set bowing_cif_first16
```

Reproduce inference/eval:

```bash
bash outputs/static_mixed_qat/20260618_pr3_clipping_bias_sweep/run_inference_eval.sh
```

## Conversion result

Checkpoint: `outputs/static_mixed_qat/20260618_pr3_clipping_bias_sweep/checkpoints/static_mixed_top60_mse.safetensors`

Conversion summary:

- converted: 306
- fallback: 0
- fp16_skip: 0
- a8w8: 122
- a16w8: 184
- activation_qdq_mode: `static_tensor_symmetric`
- static_clipping: `mse`
- enable_bias_correction: true
- Wan VAE: not quantized; DiT Linear only

## Eval result

Eval: bowing_cif first 16 frames, same-run FP16 output as reference.

- PR2 top60 minmax baseline: 15.5816900220 dB vs FP16
- PR3 top60 mse + bias correction: 15.3125204370 dB vs FP16

Conclusion: this particular `mse + bias_correction` combination did not improve the PR2 top60 minmax result on the smoke clip. Keep the implementation because it is useful for ablation/reproducibility, but do not promote it as default. For PR4, prioritize observer/freeze QAT to adapt the static student to fixed qparams rather than relying on cache-only clipping alone.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/scripts/ptq/test_static_clipping_policy.py \
  tests/scripts/experiments/test_static_mixed_policy_sweep.py \
  tests/scripts/ptq/test_lsgquant_convert_policy.py \
  tests/test_fakequant_august_recovery.py -q
# 15 passed in 3.34s

.venv/bin/python -m pytest \
  tests/test_output_qdq.py \
  tests/test_qat_pipeline.py \
  tests/test_fakequant_august_recovery.py \
  tests/test_static_ptq_baseline.py \
  tests/test_lsgquant_draq_static.py \
  tests/scripts/ptq/test_lsgquant_volts_policy.py \
  tests/scripts/ptq/test_static_clipping_policy.py \
  tests/scripts/experiments/test_static_mixed_policy_sweep.py \
  tests/scripts/ptq/test_lsgquant_convert_policy.py -q
# 70 passed in 6.55s
```
