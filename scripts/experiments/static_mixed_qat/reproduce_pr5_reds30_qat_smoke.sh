#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="outputs/static_mixed_qat/20260618_pr5_reds30_qat"
POLICY="$OUT/static_mixed_policy.json"
.venv/bin/python scripts/qat/make_static_mixed_policy.py --output "$POLICY"
.venv/bin/python -u scripts/qat/run_september_video_qat_eval.py \
  --run_id static_mixed_observer_freeze_reds30_smoke_20260618_pr5 \
  --output_root "$OUT" \
  --train_video_dir /home/user/data/REDs/REDS30_videos \
  --max_train_videos 30 \
  --prepare_frames 4 \
  --latent_size 16x16 \
  --policy_json "$POLICY" \
  --activation_qdq_mode dynamic_asymmetric \
  --observer_freeze_step 1 \
  --smoke_steps 1 \
  --ema_decay 0.0 \
  --smoke
.venv/bin/python scripts/experiments/static_mixed_qat/update_leaderboard.py \
  --leaderboard outputs/static_mixed_qat/leaderboard.jsonl \
  --run_id 20260618_pr5_static_mixed_reds30_observer_freeze_smoke \
  --policy "$POLICY" \
  --reproduce_script scripts/experiments/static_mixed_qat/reproduce_pr5_reds30_qat_smoke.sh \
  --checkpoint "$OUT/static_mixed_observer_freeze_reds30_smoke_20260618_pr5/train/flashvsr_v1.1_qat_fakequant.pt" \
  --manifest "$OUT/static_mixed_observer_freeze_reds30_smoke_20260618_pr5/samples/manifest.jsonl" \
  --a8_layers 240 \
  --a16_layers 66 \
  --activation_qdq_mode mixed_policy_static_asymmetric \
  --clipping observer_freeze \
  --qat \
  --observer observer_freeze \
  --observer_steps 1 \
  --freeze_step 1 \
  --total_steps 1 \
  --eval_set REDS30-training-smoke \
  --notes "training_dataset=REDS30; max_train_videos=30; smoke_steps=1; pseudo_latent; deterministic_context; Wan VAE unquantized"
.venv/bin/python scripts/experiments/static_mixed_qat/render_leaderboard.py \
  --leaderboard outputs/static_mixed_qat/leaderboard.jsonl \
  --output outputs/static_mixed_qat/leaderboard.html
.venv/bin/python scripts/experiments/static_mixed_qat/build_npu_handoff_package.py \
  --leaderboard outputs/static_mixed_qat/leaderboard.jsonl \
  --leaderboard_html outputs/static_mixed_qat/leaderboard.html \
  --gt_dir /home/user/data/REDs/REDS30_videos \
  --out_dir outputs/static_mixed_qat/20260618_pr5_npu_handoff \
  --sync_daily_dir /home/user/SynologyDrive/daily \
  --daily_prefix 20260618
