#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
RUN="outputs/static_mixed_qat/20260618_pr2_policy_sweep_top80_90"
EVAL="$RUN/eval_bowing_first16"
mkdir -p "$EVAL/logs" "$EVAL/videos" "$EVAL/psnr"
.venv/bin/python -u scripts/experiments/static_mixed_qat/run_policy_sweep.py \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  --calibration_cache outputs/2026-06-12_a8w8_tensor_sym/calib_cache.json \
  --sensitivity_json outputs/true_sensitivity/sensitivity_per_layer.json \
  --out_dir "$RUN" \
  --a16_percent 80,90 \
  --activation_qdq_mode static_tensor_symmetric \
  --clipping minmax \
  --leaderboard outputs/static_mixed_qat/leaderboard.jsonl \
  --eval_set convert_only
INPUT="data/lowres/bowing_cif.mp4"
COMMON=(--input "$INPUT" --scale 4 --device cuda:0 --mode tiny --vae_model Wan2.1 --tiled_vae --tiled_dit --tile_size 256 --tile_overlap 24 --frame_chunk_size 16 --end_frame 16 --precision bf16 --no_color_fix)
FP16="outputs/static_mixed_qat/20260618_pr2_policy_sweep/eval_bowing_first16/videos/fp16_bowing_first16.mp4"
if [[ ! -s "$FP16" ]]; then
  FP16="$EVAL/videos/fp16_bowing_first16.mp4"
  .venv/bin/python -u cli_main.py "${COMMON[@]}" --output "$FP16" --quantize_mode None > "$EVAL/logs/fp16.log" 2>&1
fi
for pct in 80 90; do
  CKPT="$RUN/checkpoints/static_mixed_top${pct}_minmax.safetensors"
  OUT="$EVAL/videos/static_mixed_top${pct}_bowing_first16.mp4"
  PSNR="$EVAL/psnr/top${pct}_vs_fp16.json"
  if [[ ! -s "$OUT" ]]; then
    .venv/bin/python -u cli_main.py "${COMMON[@]}" --output "$OUT" --quantize_mode FakeQuant_A8W8 --ckpt_path "$CKPT" > "$EVAL/logs/top${pct}.log" 2>&1
  fi
  .venv/bin/python scripts/compare_video_psnr.py "$FP16" "$OUT" --out-json "$PSNR" > "$EVAL/logs/top${pct}_psnr.log" 2>&1
  A8=$(python - <<PY
import json
p=json.load(open('$RUN/policies/mixed_top${pct}_a16.json'))
print(p['counts']['a8w8'])
PY
)
  A16=$(python - <<PY
import json
p=json.load(open('$RUN/policies/mixed_top${pct}_a16.json'))
print(p['counts']['a16w8'])
PY
)
  .venv/bin/python scripts/experiments/static_mixed_qat/update_leaderboard.py \
    --leaderboard outputs/static_mixed_qat/leaderboard.jsonl \
    --run_id "20260618_pr2_top${pct}_bowing_first16" \
    --policy "$RUN/policies/mixed_top${pct}_a16.json" \
    --reproduce_script scripts/experiments/static_mixed_qat/reproduce_pr2_top80_90_eval.sh \
    --checkpoint "$CKPT" \
    --manifest "$RUN/manifest.json" \
    --psnr_json "$PSNR" \
    --a8_layers "$A8" \
    --a16_layers "$A16" \
    --activation_qdq_mode static_tensor_symmetric \
    --clipping minmax \
    --eval_set bowing_cif_first16 \
    --notes "PR2 top${pct} real inference eval vs same-run FP16"
done
.venv/bin/python scripts/experiments/static_mixed_qat/render_leaderboard.py --leaderboard outputs/static_mixed_qat/leaderboard.jsonl --output outputs/static_mixed_qat/leaderboard.html
.venv/bin/python scripts/experiments/static_mixed_qat/build_npu_handoff_package.py \
  --leaderboard outputs/static_mixed_qat/leaderboard.jsonl \
  --leaderboard_html outputs/static_mixed_qat/leaderboard.html \
  --gt_dir /home/user/data/REDs/REDS30_videos \
  --out_dir outputs/static_mixed_qat/20260618_pr5_npu_handoff \
  --sync_daily_dir /home/user/SynologyDrive/daily \
  --daily_prefix 20260618
