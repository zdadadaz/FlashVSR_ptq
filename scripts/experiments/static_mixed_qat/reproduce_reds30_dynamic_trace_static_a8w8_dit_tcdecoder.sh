#!/usr/bin/env bash
# REDS30 dynamic-trace static A8W8 calibration for DiT + TCDecoder, then REDS4/000 eval vs FP16.
# Contract:
#   - DiT Linear: dynamic_asymmetric A8W8 checkpoint is replayed on REDS30, then frozen as static_token_asymmetric.
#   - TCDecoder Conv2d/Conv3d/Linear: hooked during the same dynamic trace and applied as static extra-op A8W8 at eval.
#   - Wan VAE remains unquantized.
set -euo pipefail

ROOT="${ROOT:-/home/user/apps/FlashVSRptq/FlashVSR_Integrated}"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
FP_CKPT="${FP_CKPT:-models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors}"
REDS30_GLOB="${REDS30_GLOB:-/home/user/data/REDs/REDS30_videos/LQ/*.mp4}"
REDS4_000_INPUT="${REDS4_000_INPUT:-/home/user/data/REDs/REDS30_videos/LQ/000.mp4}"
FRAMES="${FRAMES:-16}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-outputs/static_mixed_qat/${STAMP}_reds30_dynamic_trace_static_a8w8_dit_tcdecoder}"
LEADERBOARD="${LEADERBOARD:-outputs/static_mixed_qat/leaderboard.jsonl}"
DAILY_DIR="${DAILY_DIR:-/home/user/SynologyDrive/daily}"
REPRO_SCRIPT="scripts/experiments/static_mixed_qat/reproduce_reds30_dynamic_trace_static_a8w8_dit_tcdecoder.sh"

mkdir -p "$BASE_OUT/dynamic_asym" "$BASE_OUT/static_token" "$BASE_OUT/tcdecoder_static" "$BASE_OUT/eval"
printf '%s\n' "$STAMP" > "$BASE_OUT/STAMP"

"$PY" - <<PY
import json
from pathlib import Path
import torch.nn as nn
from scripts.ptq.fakequant_convert import build_dit
base = Path('$BASE_OUT')
layer_names = [name for name, module in build_dit().named_modules() if isinstance(module, nn.Linear)]
if len(layer_names) != 306:
    raise SystemExit(f'Expected 306 Linear layers, got {len(layer_names)}')
for label, qdq, note in [
    ('dynamic_asym', 'dynamic_asymmetric', 'all 306 DiT Linear A8W8; dynamic model for calibration trace'),
    ('static_token', 'static_token_asymmetric', 'all 306 DiT Linear A8W8; frozen per-token qparams captured from REDS30 dynamic quantized CLI trajectory; TCDecoder static A8W8 applied via extra-op cache at eval'),
]:
    policy = {
        'schema_version': 'flashvsr.reds30.dynamic_trace_static_token.v1',
        'quant_scope': 'dit_linear_plus_tcdecoder_extra_ops' if label == 'static_token' else 'dit_linear_only',
        'wan_vae_quantized': False,
        'tcdecoder_quantized': label == 'static_token',
        'tcdecoder_activation_qdq_mode': 'static_tensor_symmetric/static_token_asymmetric' if label == 'static_token' else None,
        'activation_qdq_mode': qdq,
        'static_ablation_label': 'reds30_dynamic_trace_static_a8w8_dit_tcdecoder',
        'counts': {'a8w8': len(layer_names), 'a16w8': 0, 'fp16_skip': 0, 'a4w4': 0},
        'layers': {name: {'mode': 'a8w8', 'activation_qdq_mode': qdq, 'reason': note} for name in layer_names},
        'notes': note,
    }
    out = base / label / 'policy.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2, sort_keys=True))
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
  --extra_calibration_cache_out "$BASE_OUT/tcdecoder_static/calib_cache_reds30_dynamic_trace.json" \
  --extra_calibration_scopes tcdecoder \
  --frames "$FRAMES" \
  --checkpoint "$BASE_OUT/dynamic_asym/checkpoint.safetensors" \
  --trace_quantize_mode FakeQuant_A8W8 \
  --mode tiny \
  --scale 4 \
  --precision bf16 \
  --calibration_granularity per_token \
  --tiled_vae --tiled_dit --tile_size 256 --tile_overlap 24 \
  --attention_mode sparse_sage_attention --sparse_ratio 2.0 --kv_ratio 3.0 --local_range 9 \
  > "$BASE_OUT/01_calibrate_reds30_dynamic_trace_dit_tcdecoder.log" 2>&1

"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --calibration_cache "$BASE_OUT/static_token/calib_cache_reds30_dynamic_trace.json" \
  --output "$BASE_OUT/static_token/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode static_token_asymmetric \
  --policy "$BASE_OUT/static_token/policy.json" \
  > "$BASE_OUT/02_convert_static_token.log" 2>&1

RUN_ID="${STAMP}_reds30_static_a8w8_dit_tcdecoder_reds4_000_first16"
"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id "$RUN_ID" \
  --policy "$BASE_OUT/static_token/policy.json" \
  --checkpoint "$BASE_OUT/static_token/checkpoint.safetensors" \
  --input_video "$REDS4_000_INPUT" \
  --frames "$FRAMES" \
  --eval_set "REDS4_000_first16_video_vs_fp16_static_dit_tcdecoder" \
  --clipping_method none \
  --teacher_ft_steps 0 \
  --static_ablation_label reds30_dynamic_trace_static_a8w8_dit_tcdecoder \
  --fakequant_extra_scopes tcdecoder \
  --fakequant_extra_calibration_cache "$BASE_OUT/tcdecoder_static/calib_cache_reds30_dynamic_trace.json" \
  --fakequant_extra_activation_qdq_mode static_tensor_symmetric \
  --reproduce_script "$REPRO_SCRIPT" \
  --leaderboard "$LEADERBOARD" \
  --daily_dir "$DAILY_DIR" \
  > "$BASE_OUT/03_eval_reds4_000_static_dit_tcdecoder.log" 2>&1

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
    'stamp': '$STAMP',
    'base_out': str(base),
    'dit_calibration_cache': str(base / 'static_token/calib_cache_reds30_dynamic_trace.json'),
    'tcdecoder_calibration_cache': str(base / 'tcdecoder_static/calib_cache_reds30_dynamic_trace.json'),
    'static_dit_checkpoint': str(base / 'static_token/checkpoint.safetensors'),
    'policy': str(base / 'static_token/policy.json'),
    'reds4_run_id': '$RUN_ID',
    'reds4_eval_dir': f'outputs/qbasicvsr/eval/$RUN_ID',
    'reds4_psnr_json': f'outputs/qbasicvsr/eval/$RUN_ID/psnr.json',
    'leaderboard': '$LEADERBOARD',
    'leaderboard_html': '${LEADERBOARD%.jsonl}.html',
}
(base / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
