# PR1 Static Mixed QAT Foundation — Execution Log

Date: 2026-06-18 17:12+
Branch: feat/static-mixed-qparams-pr1

## Scope

Implemented the first PR from `20260618_171210_flashvsr_static_mixed_a8w8_a16w8_qat_plan`:

- Static mixed A8W8/A16W8 policy generator from per-layer sensitivity reports.
- Policy schema `flashvsr.static_mixed_policy.v1` with DiT-only scope and `wan_vae_quantized=false`.
- A16 fallback means `a16w8`: A16 activation passthrough while W8 weights remain quantized.
- Added `static_tensor_symmetric` and static-DRAQ modes to policy validation allow-list.
- Added JSONL leaderboard writer and HTML renderer.
- Added CPU/schema tests.

This PR also includes already-present foundational output-QDQ/static calibration edits in the dirty worktree, verified by the existing QDQ/PTQ/QAT tests.

## Validation

Commands run:

```bash
.venv/bin/python -m pytest \
  tests/scripts/ptq/test_static_mixed_policy_generator.py \
  tests/scripts/experiments/test_static_mixed_leaderboard.py \
  tests/test_output_qdq.py \
  tests/test_qat_pipeline.py \
  tests/test_fakequant_august_recovery.py \
  tests/test_static_ptq_baseline.py \
  tests/test_lsgquant_draq_static.py \
  tests/scripts/ptq/test_lsgquant_volts_policy.py -q
```

Result: `65 passed in 3.86s`.

Execution smoke:

```bash
.venv/bin/python scripts/ptq/build_static_mixed_policy.py \
  --sensitivity_json outputs/static_mixed_qat/20260618_pr1_smoke/sensitivity/per_layer_mse.json \
  --a16_percent 10,20,40,60 \
  --output_dir outputs/static_mixed_qat/20260618_pr1_smoke/policies

.venv/bin/python scripts/experiments/static_mixed_qat/update_leaderboard.py ...
.venv/bin/python scripts/experiments/static_mixed_qat/render_leaderboard.py ...
```

Result: generated top10/top20/top40/top60 policies and rendered `outputs/static_mixed_qat/leaderboard.html`.

## Notes

PR1 is schema/tooling only. Real calibration, inference and PSNR sweep are PR2.
