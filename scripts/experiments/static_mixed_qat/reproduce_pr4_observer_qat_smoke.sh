#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="outputs/static_mixed_qat/20260618_pr4_observer_qat"
.venv/bin/python scripts/qat/make_static_mixed_policy.py --output "$OUT/static_mixed_policy.json"
.venv/bin/python -u scripts/qat/run_september_video_qat_eval.py \
  --run_id static_mixed_observer_freeze_smoke4f_20260618_pr4 \
  --output_root "$OUT" \
  --max_train_videos 1 \
  --prepare_frames 4 \
  --latent_size 16x16 \
  --steps 1 \
  --smoke_steps 1 \
  --smoke \
  --policy_json "$OUT/static_mixed_policy.json" \
  --activation_qdq_mode dynamic_asymmetric \
  --observer_freeze_step 1 \
  --observer_ema_decay 0.95 \
  --ema_decay 0
