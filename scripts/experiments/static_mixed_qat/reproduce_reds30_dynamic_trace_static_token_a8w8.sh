#!/usr/bin/env bash
# REDS30 dynamic-trace static-token A8W8 calibration + bowing / REDS4-000 eval.
# Supported static PTQ contract:
#   1) Convert FP DiT -> dynamic_asymmetric FakeQuant A8W8 checkpoint.
#   2) Replay the real CLI/nodes inference path on REDS30 LQ clips with that dynamic checkpoint.
#   3) Freeze the hooked dynamic quantized trajectory qparams as static_token_asymmetric.
#   4) Evaluate with CLI-aligned inference; Wan VAE remains unquantized, DiT Linear only is quantized.
set -euo pipefail

ROOT="${ROOT:-/home/user/apps/FlashVSRptq/FlashVSR_Integrated}"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
FP_CKPT="${FP_CKPT:-models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors}"
REDS30_GLOB="${REDS30_GLOB:-/home/user/data/REDs/REDS30_videos/LQ/*.mp4}"
BOWING_INPUT="${BOWING_INPUT:-data/lowres/bowing_cif.mp4}"
REDS4_000_INPUT="${REDS4_000_INPUT:-/home/user/data/REDs/REDS30_videos/LQ/000.mp4}"
FRAMES="${FRAMES:-16}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-outputs/static_mixed_qat/${STAMP}_reds30_dynamic_trace_static_token_a8w8}"
LEADERBOARD="${LEADERBOARD:-outputs/static_mixed_qat/leaderboard.jsonl}"
REPRO_SCRIPT="scripts/experiments/static_mixed_qat/reproduce_reds30_dynamic_trace_static_token_a8w8.sh"
DAILY_DIR="${DAILY_DIR:-/home/user/SynologyDrive/daily}"

mkdir -p "$BASE_OUT/dynamic_asym" "$BASE_OUT/static_token" "$BASE_OUT/eval"

"$PY" - <<PY
import json
from pathlib import Path
import torch.nn as nn
from scripts.ptq.fakequant_convert import build_dit

base = Path('$BASE_OUT')
layer_names = [name for name, module in build_dit().named_modules() if isinstance(module, nn.Linear)]
if len(layer_names) != 306:
    raise SystemExit(f"Expected 306 Linear layers, got {len(layer_names)}")

def write_policy(label, qdq_mode, note):
    layers = {
        name: {'mode': 'a8w8', 'activation_qdq_mode': qdq_mode, 'reason': note}
        for name in layer_names
    }
    policy = {
        'schema_version': 'flashvsr.reds30.dynamic_trace_static_token.v1',
        'quant_scope': 'dit_linear_only',
        'wan_vae_quantized': False,
        'activation_qdq_mode': qdq_mode,
        'static_ablation_label': label,
        'counts': {'a8w8': len(layer_names), 'a16w8': 0, 'fp16_skip': 0, 'a4w4': 0},
        'layers': layers,
        'notes': note,
    }
    out = base / label / 'policy.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2, sort_keys=True))

write_policy('dynamic_asym', 'dynamic_asymmetric', 'all 306 DiT Linear A8W8; activation qparams computed online per token')
write_policy('static_token', 'static_token_asymmetric', 'all 306 DiT Linear A8W8; frozen per-token qparams captured from REDS30 dynamic quantized CLI trajectory')
PY

"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --output "$BASE_OUT/dynamic_asym/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode dynamic_asymmetric \
  --policy "$BASE_OUT/dynamic_asym/policy.json" \
  > "$BASE_OUT/00_convert_dynamic_asym.log" 2>&1

"$PY" -u scripts/ptq/qbasicvsr_oracle_trace_calibrate.py \
  --input_glob "$REDS30_GLOB" \
  --max_inputs 30 \
  --output_cache "$BASE_OUT/static_token/calib_cache_reds30_dynamic_trace.json" \
  --frames "$FRAMES" \
  --checkpoint "$BASE_OUT/dynamic_asym/checkpoint.safetensors" \
  --trace_quantize_mode FakeQuant_A8W8 \
  --mode tiny \
  --scale 4 \
  --precision bf16 \
  --calibration_granularity per_token \
  --tiled_vae --tiled_dit --tile_size 256 --tile_overlap 24 \
  --attention_mode sparse_sage_attention --sparse_ratio 2.0 --kv_ratio 3.0 --local_range 9 \
  > "$BASE_OUT/01_calibrate_reds30_dynamic_trace.log" 2>&1

"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --calibration_cache "$BASE_OUT/static_token/calib_cache_reds30_dynamic_trace.json" \
  --output "$BASE_OUT/static_token/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode static_token_asymmetric \
  --policy "$BASE_OUT/static_token/policy.json" \
  > "$BASE_OUT/02_convert_static_token.log" 2>&1

"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id "${STAMP}_reds30_static_token_bowing_first16" \
  --policy "$BASE_OUT/static_token/policy.json" \
  --checkpoint "$BASE_OUT/static_token/checkpoint.safetensors" \
  --input_video "$BOWING_INPUT" --frames "$FRAMES" --eval_set "bowing_cif_first16_video_vs_fp16" \
  --clipping_method none --teacher_ft_steps 0 --static_ablation_label reds30_dynamic_trace_static_token_a8w8 \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD" --daily_dir "$DAILY_DIR" \
  > "$BASE_OUT/03_eval_bowing.log" 2>&1

BOWING_FP16="outputs/qbasicvsr/eval/${STAMP}_reds30_static_token_bowing_first16/fp16.mp4"

"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id "${STAMP}_reds30_static_token_reds4_000_first16" \
  --policy "$BASE_OUT/static_token/policy.json" \
  --checkpoint "$BASE_OUT/static_token/checkpoint.safetensors" \
  --input_video "$REDS4_000_INPUT" --frames "$FRAMES" --eval_set "REDS4_000_first16_video_vs_fp16" \
  --clipping_method none --teacher_ft_steps 0 --static_ablation_label reds30_dynamic_trace_static_token_a8w8 \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD" --daily_dir "$DAILY_DIR" \
  > "$BASE_OUT/04_eval_reds4_000.log" 2>&1

"$PY" scripts/experiments/static_mixed_qat/render_leaderboard.py \
  --leaderboard "$LEADERBOARD" \
  --output "${LEADERBOARD%.jsonl}.html"

PREFIX="$(date +%Y%m%d)"
cp "$LEADERBOARD" "$DAILY_DIR/${PREFIX}_flashvsr_static_mixed_leaderboard.jsonl"
cp "${LEADERBOARD%.jsonl}.html" "$DAILY_DIR/${PREFIX}_flashvsr_static_mixed_leaderboard.html"

"$PY" - <<PY
import json
from pathlib import Path
base = Path('$BASE_OUT')
summary = {
    'base_out': str(base),
    'calibration_cache': str(base / 'static_token/calib_cache_reds30_dynamic_trace.json'),
    'static_checkpoint': str(base / 'static_token/checkpoint.safetensors'),
    'bowing_psnr': 'outputs/qbasicvsr/eval/${STAMP}_reds30_static_token_bowing_first16/psnr.json',
    'reds4_000_psnr': 'outputs/qbasicvsr/eval/${STAMP}_reds30_static_token_reds4_000_first16/psnr.json',
    'leaderboard': '$LEADERBOARD',
}
(base / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
