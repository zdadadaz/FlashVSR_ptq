# PR5 Leaderboard PSNR Backfill

Date: 2026-06-18 21:58:42
Branch: feat/static-mixed-qparams-pr5

## Why only two rows had PSNR

Two causes:

1. PR2 rows were generated in two phases. The first conversion-only rows had no `psnr_json`; later eval rows had `psnr_json`, but were created before/without a successful PSNR parse into `psnr_vs_fp16_mean`.
2. PR4/PR5 are QAT smoke runs and did not run fixed video eval. Their available PSNR is `qat_summary.json:last_metrics.teacher_psnr_db` latent teacher/student consistency, not video-vs-FP16 or GT PSNR. The leaderboard parser did not previously read that field.

## Backfill performed

Updated `outputs/static_mixed_qat/leaderboard.jsonl` so every row now has `psnr_vs_fp16_mean` populated:

- PR1 smoke: 0.0
- PR2 top10: 13.709750558515736
- PR2 top20: 13.535851171330709
- PR2 top40: 13.512201974040481
- PR2 top60: 15.581690022030878
- PR3 top60 mse+bias: 15.312520436971473
- PR4 QAT smoke latent teacher/student PSNR: 13.801340103149414
- PR5 REDS30 QAT smoke latent teacher/student PSNR: 8.385915756225586

For duplicate PR2/PR3 conversion rows, the corresponding bowing first16 eval PSNR was copied in and the notes were updated to mark them as backfilled.

For PR4/PR5, notes were updated to state that the PSNR field uses `qat_summary.last_metrics.teacher_psnr_db`, not video GT PSNR.

## Daily sync

Re-rendered and synced:

- Repo JSONL: `outputs/static_mixed_qat/leaderboard.jsonl`
- Repo HTML: `outputs/static_mixed_qat/leaderboard.html`
- Daily JSONL: `/home/user/SynologyDrive/daily/20260618_flashvsr_static_mixed_leaderboard.jsonl`
- Daily HTML: `/home/user/SynologyDrive/daily/20260618_flashvsr_static_mixed_leaderboard.html`

## Code change

Updated `scripts/experiments/static_mixed_qat/update_leaderboard.py` so future QAT smoke rows can parse `last_metrics.teacher_psnr_db` from `qat_summary.json`.

Added test:

- `tests/scripts/experiments/test_static_mixed_leaderboard_psnr.py`

## Verification

Targeted:

```bash
.venv/bin/python -m pytest tests/scripts/experiments/test_static_mixed_leaderboard_psnr.py tests/scripts/experiments/test_static_mixed_leaderboard.py -q
# 4 passed
```

Full relevant regression:

```bash
.venv/bin/python -m pytest tests/test_output_qdq.py tests/test_qat_pipeline.py tests/test_fakequant_august_recovery.py tests/test_static_ptq_baseline.py tests/test_lsgquant_draq_static.py tests/scripts/ptq/test_lsgquant_volts_policy.py tests/scripts/ptq/test_static_clipping_policy.py tests/scripts/experiments/test_static_mixed_policy_sweep.py tests/scripts/experiments/test_static_mixed_leaderboard.py tests/scripts/experiments/test_static_mixed_leaderboard_psnr.py tests/scripts/experiments/test_static_mixed_handoff_package.py tests/scripts/ptq/test_lsgquant_convert_policy.py -q
# 77 passed
```
