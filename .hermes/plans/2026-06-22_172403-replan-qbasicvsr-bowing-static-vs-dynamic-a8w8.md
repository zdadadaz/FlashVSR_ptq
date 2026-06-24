# Replan: QBasicVSR bowing static A8W8 calibration vs pure dynamic asymmetric A8W8

## Correction

Previous run compared the wrong thing for the user's intended experiment.

Mistakes to correct:

1. The `dynamic_asym` row inherited an existing QBasicVSR policy with A16 fallback layers (`activation_mode_code {2:300, 1:6}`), so it was not a pure all-Linear A8W8 baseline.
2. The `oracle_trace_static` row was framed as an upper-bound experiment but was not the intended paper-style comparison: static A8W8 calibrated on the same bowing video should be compared directly against pure dynamic asymmetric A8W8 under the same layer scope and eval path.
3. The new experiment must make the dynamic and static checkpoints differ only in activation qparam source:
   - dynamic: activation qparams computed online per activation tensor/token
   - static: activation qparams calibrated from bowing video and frozen before eval

## Goal

Run a fair bowing-first16 comparison:

- **A. Pure dynamic asymmetric A8W8**
  - DiT Linear scope only
  - all 306 WanVideoDiT `Linear` layers use A8 activation + W8 weight
  - `activation_qdq_mode=dynamic_asymmetric`
  - no calibration cache
  - no A16 fallback, no mixed policy

- **B. Pure static calibrated A8W8 on bowing**
  - same DiT Linear scope
  - all 306 WanVideoDiT `Linear` layers use A8 activation + W8 weight
  - `activation_qdq_mode=static_asymmetric`
  - calibration cache collected from the *same* `data/lowres/bowing_cif.mp4` first16 path used for eval
  - no A16 fallback, no HQ-VSR union, no OMSE/teacher clipping/fine-tune for the main comparison

Expected: these two should be close if static qparams are collected/applied with the correct path, granularity, and full layer coverage.

## Experimental contract

### Fixed settings

- Repo: `/home/user/apps/FlashVSRptq/FlashVSR_Integrated`
- Python: `.venv/bin/python`
- Input: `data/lowres/bowing_cif.mp4`
- Frames: first 16
- Eval reference: same-run or reused FP16 video from the exact same CLI settings
- Quantized scope: WanVideoDiT `Linear` only
- Unquantized: Wan VAE, TCDecoder, LQ conv/proj, Conv3d patch embedding, norms, softmax/matmuls, scheduler/video I/O
- Eval entrypoint: `scripts/ptq/run_qbasicvsr_temporal_eval.py` or direct `cli_main.py`, but both rows must use the same path
- Leaderboard: `outputs/static_mixed_qat/leaderboard.jsonl` plus daily sync

### Purity checks required before accepting a row

For both checkpoints:

```python
activation_mode_code == {2: 306}
```

For dynamic:

```python
activation_qdq_mode == {2: 306}
```

For static:

```python
activation_qdq_mode == {0: 306}
cache layer coverage == 306 / 306
no missing act_scale / zero_point
```

If any row shows `activation_mode_code {2:300, 1:6}` or similar, reject it as mixed, not pure.

## Proposed implementation plan

### Step 1 — Create pure all-A8 policy generator

Add or reuse a tiny helper that emits policies without inheriting the old smoke mixed policy:

- `outputs/qbasicvsr/bowing_pure_a8w8/dynamic_asym/policy.json`
- `outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/policy.json`

Policy content:

- `counts.a8w8 = 306`
- `counts.a16w8 = 0`
- every `layers[*].mode = a8w8`
- no `fp16_skip`
- `activation_qdq_mode` set globally and per-layer
- `quant_scope = dit_linear_only`
- `wan_vae_quantized = false`

Validation: inspect converted checkpoint buffers, not just the policy JSON.

### Step 2 — Convert pure dynamic asymmetric checkpoint

Command shape:

```bash
.venv/bin/python scripts/ptq/fakequant_convert.py \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  --output outputs/qbasicvsr/bowing_pure_a8w8/dynamic_asym/checkpoint.safetensors \
  --mode a8w8 \
  --activation_qdq_mode dynamic_asymmetric \
  --policy outputs/qbasicvsr/bowing_pure_a8w8/dynamic_asym/policy.json
```

Acceptance:

- converted layer count: 306
- `activation_mode_code {2:306}`
- `activation_qdq_mode {2:306}`

### Step 3 — Rebuild static bowing calibration using the same eval path

Do not use random/proxy frame-only calibration if the intended experiment is target-video calibration.

Use a calibration script that runs the same FlashVSR/QBasicVSR eval path and hooks every DiT `Linear` input during bowing first16.

Current candidate:

- `scripts/ptq/qbasicvsr_oracle_trace_calibrate.py`

But before trusting it, add/check diagnostics:

1. Confirm the script uses identical `cli_main.py` settings as the eval row:
   - `--mode tiny`
   - `--scale 4`
   - same `--end_frame/frames=16`
   - same precision
   - same tiling/offload flags as eval
2. Confirm it hooks the exact modules that conversion names use.
3. Confirm raw qparams have sane shapes:
   - per-channel static asymmetric: `[in_features]` for each layer
   - no accidental extra leading call dimension after cat/stack
4. Confirm no cache key prefix mismatch between hooked names and checkpoint conversion names.
5. Confirm static QDQ replay on the captured activations has low MSE and low saturation before running full video eval.

Static calibration command shape:

```bash
.venv/bin/python -u scripts/ptq/qbasicvsr_oracle_trace_calibrate.py \
  --input data/lowres/bowing_cif.mp4 \
  --output_cache outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/calib_cache.json \
  --output_video outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/fp16_trace.mp4 \
  --frames 16 \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  > outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/calibrate.log 2>&1
```

### Step 4 — Static QDQ sanity diagnostic before conversion/eval

Add a small diagnostic script or mode if missing:

- Input: calibration cache + saved/streamed activation samples from bowing first16
- For every layer, compute:
  - dynamic asymmetric activation QDQ MSE
  - static asymmetric activation QDQ MSE using calibrated qparams
  - saturation / clipping rate under static qparams
  - SQNR gap dynamic vs static

Acceptance gate before video eval:

- 306/306 layers checked
- static saturation near zero on the calibration/eval trace
- static activation MSE same order as dynamic; if static is 10x+ worse before weights/DiT propagation, the qparam construction/broadcast/granularity is wrong and video PSNR should not be trusted

This diagnostic is the key difference from the previous mistaken run: do not wait until final PSNR to discover static qparams are broken.

### Step 5 — Convert pure static bowing checkpoint

```bash
.venv/bin/python scripts/ptq/fakequant_convert.py \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  --calibration_cache outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/calib_cache.json \
  --output outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/checkpoint.safetensors \
  --mode a8w8 \
  --activation_qdq_mode static_asymmetric \
  --policy outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/policy.json
```

Acceptance:

- `activation_mode_code {2:306}`
- `activation_qdq_mode {0:306}`
- all `act_scale`/`act_zero_point` loaded from cache, not default scale=1

### Step 6 — Evaluate both rows with identical FP16 reference

Run both through the same wrapper and same FP16 reference:

```bash
.venv/bin/python scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_bowing16_pure_dynamic_asym_a8w8 \
  --policy outputs/qbasicvsr/bowing_pure_a8w8/dynamic_asym/policy.json \
  --checkpoint outputs/qbasicvsr/bowing_pure_a8w8/dynamic_asym/checkpoint.safetensors \
  --input_video data/lowres/bowing_cif.mp4 \
  --frames 16 \
  --eval_set bowing_cif_first16_video_vs_fp16 \
  --fp16_video <same_fp16_video> \
  --clipping_method none \
  --teacher_ft_steps 0 \
  --static_ablation_label pure_dynamic_asym_a8w8 \
  --leaderboard outputs/static_mixed_qat/leaderboard.jsonl
```

```bash
.venv/bin/python scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_bowing16_pure_static_asym_a8w8_bowing_calib \
  --policy outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/policy.json \
  --checkpoint outputs/qbasicvsr/bowing_pure_a8w8/static_bowing/checkpoint.safetensors \
  --input_video data/lowres/bowing_cif.mp4 \
  --frames 16 \
  --eval_set bowing_cif_first16_video_vs_fp16 \
  --fp16_video <same_fp16_video> \
  --clipping_method minmax_bowing_target \
  --teacher_ft_steps 0 \
  --static_ablation_label pure_static_asym_a8w8_bowing_calib \
  --leaderboard outputs/static_mixed_qat/leaderboard.jsonl
```

### Step 7 — If static is still far from dynamic, debug in this order

Do not immediately add fallback/A16/OMSE. First prove whether the static experiment is truly equivalent to the intended calibration setup.

1. **Policy purity**: verify no hidden A16 fallback in either row.
2. **Cache/key coverage**: 306 names match exactly.
3. **Scale shape/broadcast**: static `act_scale` should broadcast over last dimension only; no accidental `[calls, C]` or `[1, calls, C]` shape.
4. **Signed int8 qparam math**:
   - `qmin=-128`, `qmax=127`
   - `scale=(max-min)/255`
   - `zp=round(qmin - min/scale).clamp(-128,127)`
5. **Activation trace mismatch**: calibration and eval must call the same path and same precision.
6. **Quantized-trajectory effect**: if FP16-trace static is clean at layer-local QDQ but video PSNR diverges, then collect qparams under fake-quant-in-loop trajectory as a second diagnostic row, clearly labelled separately.

## Leaderboard rows to add/update

Use new run IDs to avoid confusing them with the previous mistaken rows:

- `qbasicvsr_bowing16_pure_dynamic_asym_a8w8`
- `qbasicvsr_bowing16_pure_static_asym_a8w8_bowing_calib`

Optional diagnostic rows only after the main pair:

- `qbasicvsr_bowing16_static_asym_a8w8_bowing_calib_quantized_trace`
- `qbasicvsr_bowing16_static_asym_a8w8_bowing_calib_per_tensor`

The previous rows should remain in history but be marked in the daily report as rejected/invalid for this specific comparison because they were mixed-policy or wrong-protocol.

## Files likely to change

- `scripts/ptq/qbasicvsr_oracle_trace_calibrate.py`
  - rename/relabel or add a new mode for target-video static calibration
  - add qparam shape/key diagnostics
  - optionally save small activation-QDQ diagnostic summaries

- `scripts/experiments/static_mixed_qat/reproduce_qbasicvsr_bowing_pure_a8w8_static_vs_dynamic.sh`
  - new reproducible script for the corrected experiment

- `scripts/ptq/run_qbasicvsr_temporal_eval.py`
  - only if leaderboard metadata needs clearer labels for `pure_static_bowing_calib`

- `outputs/static_mixed_qat/leaderboard.jsonl`
- `outputs/static_mixed_qat/leaderboard.html`
- `/home/user/SynologyDrive/daily/YYYYMMDD_flashvsr_static_mixed_leaderboard.{jsonl,html}`
- `/home/user/SynologyDrive/daily/YYYYMMDD_qbasicvsr_bowing_pure_a8w8_static_vs_dynamic.md`

## Success criteria

Main acceptance:

- Dynamic checkpoint is pure: 306/306 A8W8, `dynamic_asymmetric`.
- Static checkpoint is pure: 306/306 A8W8, `static_asymmetric`.
- Static cache is bowing-first16 target-video calibration with 306/306 coverage.
- Both rows are evaluated against the same FP16 reference and same eval path.
- Leaderboard rows include reproduce script and checksum.

Expected quality:

- `PSNR(static_bowing_calib)` should be close to `PSNR(dynamic_asym)` for the intended QBasicVSR-style target-calibrated comparison.
- If not close, the report must include layer-local activation-QDQ diagnostics showing whether the mismatch is due to qparam construction/application versus quantized-trajectory drift.

## Reporting format

Final report should state explicitly:

- This is a corrected experiment replacing the prior mixed-policy/oracle-trace attempt.
- Quantization scope: DiT Linear only; normalization/softmax/VAE/etc. not quantized.
- Dynamic vs static differ only by activation qparam source.
- Checkpoint buffer counts.
- Calibration cache coverage.
- PSNR values and delta.
- If static is not close, include the first failing diagnostic evidence rather than speculating.
