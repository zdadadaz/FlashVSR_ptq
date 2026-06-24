#!/usr/bin/env bash
# Reproduce QBasicVSR-inspired DRAQ/static-asymmetric/OMSE/clipFT leaderboard rows.
# Scope: FlashVSR WanVideoDiT Linear only; Wan VAE remains unquantized.
set -euo pipefail

ROOT="${ROOT:-/home/user/apps/FlashVSRptq/FlashVSR_Integrated}"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"

FP_CKPT="${FP_CKPT:-models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors}"
CALIB_CACHE="${CALIB_CACHE:-outputs/2026-06-13_reds_ptq_a8w8/calib_cache.json}"
INPUT_VIDEO="${INPUT_VIDEO:-data/lowres/bowing_cif.mp4}"
FP16_VIDEO="${FP16_VIDEO:-outputs/qbasicvsr/eval/qbasicvsr_proxy_bv0_seed1_bowing16_smoke/fp16.mp4}"
DRAQ_PTQ_VIDEO="${DRAQ_PTQ_VIDEO:-outputs/qbasicvsr/eval/qbasicvsr_proxy_bv0_seed1_bowing16_smoke/ptq.mp4}"
LEADERBOARD="${LEADERBOARD:-outputs/static_mixed_qat/leaderboard.jsonl}"
REPRO_SCRIPT="scripts/experiments/static_mixed_qat/reproduce_qbasicvsr_static_ablation.sh"
BASE_OUT="outputs/qbasicvsr/static_ablation"
EVAL_SET="bowing_cif_first16_video_vs_fp16"
FRAMES="${FRAMES:-16}"

mkdir -p "$BASE_OUT" outputs/qbasicvsr/eval

# 1) DRAQ reference row, using the existing QBasicVSR temporal DRAQ checkpoint.
"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_proxy_bv0_seed1_bowing16_draq_ref \
  --policy outputs/qbasicvsr/smoke/policy.json \
  --checkpoint outputs/qbasicvsr/smoke/qbasicvsr_temporal.safetensors \
  --input_video "$INPUT_VIDEO" --frames "$FRAMES" --eval_set "$EVAL_SET" \
  --fp16_video "$FP16_VIDEO" --ptq_video "$DRAQ_PTQ_VIDEO" \
  --clipping_method none --teacher_ft_steps 0 --static_ablation_label draq_ref \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD"

# 2) Static asymmetric minmax/EMA conversion and eval.
"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --calibration_cache "$CALIB_CACHE" \
  --output "$BASE_OUT/static_asym_calib/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode static_asymmetric \
  --policy_json "$BASE_OUT/static_asym_calib/policy.json"

"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_proxy_bv0_seed1_bowing16_static_asym_calib \
  --policy "$BASE_OUT/static_asym_calib/policy.json" \
  --checkpoint "$BASE_OUT/static_asym_calib/checkpoint.safetensors" \
  --input_video "$INPUT_VIDEO" --frames "$FRAMES" --eval_set "$EVAL_SET" \
  --fp16_video "$FP16_VIDEO" \
  --clipping_method minmax_ema --teacher_ft_steps 0 --static_ablation_label static_asym_calib \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD"

# 3) Static asymmetric OMSE cache refinement, conversion, and eval.
"$PY" scripts/ptq/qbasicvsr_static_omse_clip.py \
  --input "$CALIB_CACHE" \
  --output "$BASE_OUT/omse/clip_cache.json"

"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --calibration_cache "$BASE_OUT/omse/clip_cache.json" \
  --output "$BASE_OUT/omse/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode static_asymmetric \
  --policy_json "$BASE_OUT/omse/policy.json"

"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_proxy_bv0_seed1_bowing16_static_asym_omse \
  --policy "$BASE_OUT/omse/policy.json" \
  --checkpoint "$BASE_OUT/omse/checkpoint.safetensors" \
  --input_video "$INPUT_VIDEO" --frames "$FRAMES" --eval_set "$EVAL_SET" \
  --fp16_video "$FP16_VIDEO" \
  --clipping_method omse --teacher_ft_steps 0 --static_ablation_label static_asym_omse \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD"

# 4) qparam-only teacher clipping fine-tune cache contract, conversion, and eval.
"$PY" scripts/qat/finetune_qbasicvsr_static_clipping.py \
  --input_cache "$BASE_OUT/omse/clip_cache.json" \
  --output_cache "$BASE_OUT/clipft/clip_cache.json" \
  --metrics_jsonl "$BASE_OUT/clipft/train/metrics.jsonl" \
  --steps 8 --lr 0.01 --dry_run

"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --calibration_cache "$BASE_OUT/clipft/clip_cache.json" \
  --output "$BASE_OUT/clipft/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode static_asymmetric \
  --policy_json "$BASE_OUT/clipft/policy.json"

"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_proxy_bv0_seed1_bowing16_static_asym_omse_clipft \
  --policy "$BASE_OUT/clipft/policy.json" \
  --checkpoint "$BASE_OUT/clipft/checkpoint.safetensors" \
  --input_video "$INPUT_VIDEO" --frames "$FRAMES" --eval_set "$EVAL_SET" \
  --fp16_video "$FP16_VIDEO" \
  --clipping_method omse_teacher_clipft --teacher_ft_steps 8 --static_ablation_label static_asym_omse_clipft \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD"

"$PY" scripts/experiments/static_mixed_qat/render_leaderboard.py \
  --leaderboard "$LEADERBOARD" \
  --output "${LEADERBOARD%.jsonl}.html"

PREFIX="$(date +%Y%m%d)"
cp "$LEADERBOARD" "/home/user/SynologyDrive/daily/${PREFIX}_flashvsr_static_mixed_leaderboard.jsonl"
cp "${LEADERBOARD%.jsonl}.html" "/home/user/SynologyDrive/daily/${PREFIX}_flashvsr_static_mixed_leaderboard.html"
