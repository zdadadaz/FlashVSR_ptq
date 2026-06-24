# PR4 Observer/Freeze Static Mixed QAT

Date: 2026-06-18
Branch: feat/static-mixed-qparams-pr4

## Scope

Completed PR4 observer/freeze QAT path for NPU-compatible static qparams:

- Fixed QAT observer activation logic so a mixed policy that contains `activation_qdq_mode=static_asymmetric` layers enables observers even when the top-level `--activation_qdq_mode` remains dynamic.
- Added `policy_requires_static_observer()` and a CPU regression test.
- Ran a GPU smoke QAT pass with observer warmup/freeze/export.
- Added a leaderboard row and one-click reproduction script for the QAT smoke artifact.

## Why the code fix matters

Before this PR, `finetune_fakequant_dit.py` only enabled observers when:

```python
args.activation_qdq_mode == "static_asymmetric"
```

That misses the intended mixed policy case:

- top-level default: dynamic/non-static
- policy-specific robust layers: `a8w8 + static_asymmetric`
- sensitive layers: `a16w8`

Now observer warmup/freeze is enabled if either the global mode or any policy entry requires static asymmetric qparams.

## Reproduce

```bash
bash scripts/experiments/static_mixed_qat/reproduce_pr4_observer_qat_smoke.sh
```

This generates a conservative mixed static QAT policy and runs a 1-step smoke QAT pass:

- train videos: `datasets/train`
- max videos: 1
- frames: 4
- latent: pseudo, `16x16`
- QAT steps: 1
- observer freeze step: 1
- EMA decay: 0
- top-level qdq mode: dynamic_asymmetric
- policy static layers: static_asymmetric

## Smoke output

Run dir:

`outputs/static_mixed_qat/20260618_pr4_observer_qat/static_mixed_observer_freeze_smoke4f_20260618_pr4`

Exported files:

- trainable QAT checkpoint: `train/flashvsr_v1.1_qat_trainable.pt`
- inference FakeQuant checkpoint: `train/flashvsr_v1.1_qat_fakequant.pt`
- summary: `train/qat_summary.json`

Observer summary from `qat_summary.json`:

- observer enabled: true
- freeze_step: 1
- ema_decay: 0.95
- student frozen: 240
- student skipped_uninitialized: 0
- student non_qat: 66
- export_source_final_freeze frozen: 240
- export_source_final_freeze skipped_uninitialized: 0
- export_source_final_freeze non_qat: 66

Last smoke metrics:

- loss: 0.0423442908
- distill_loss: 0.0416740775
- temporal_loss: 0.0134042986
- teacher_psnr_db: 13.8013401031

## Leaderboard

Added row:

`20260618_pr4_static_mixed_observer_freeze_smoke`

Fields:

- a8_layers: 240
- a16_layers: 66
- qat: true
- observer: `ema_minmax`
- observer_steps/freeze_step/total_steps: 1/1/1
- eval_set: `qat_smoke_latent_4f`

## Validation

```bash
.venv/bin/python -m pytest \
  tests/test_qat_pipeline.py \
  tests/scripts/ptq/test_static_clipping_policy.py \
  tests/scripts/experiments/test_static_mixed_policy_sweep.py -q
# 26 passed in 1.15s
```

## Notes

This is a smoke QAT/export validation, not a full PSNR-quality QAT run. Full run should use real VAE/context manifest when available and evaluate GT PSNR drop. Current repository dataset discovery still indicates no paired LQ/GT layout in `datasets/train` unless the user supplies a paired dataset path.
