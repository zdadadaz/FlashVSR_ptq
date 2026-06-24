#!/usr/bin/env bash
# Reproduce bowing-only pure all-Linear A8W8 dynamic vs aligned static-token rows.
# Correct static calibration contract:
#   1) create dynamic_asymmetric A8W8 checkpoint
#   2) trace the real CLI inference path with that dynamic checkpoint
#   3) convert static_token_asymmetric from the dynamic-path qparams
#   4) evaluate dynamic and static with identical CLI settings
# Scope: FlashVSR WanVideoDiT Linear only; Wan VAE remains unquantized.
set -euo pipefail

ROOT="${ROOT:-/home/user/apps/FlashVSRptq/FlashVSR_Integrated}"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
FP_CKPT="${FP_CKPT:-models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors}"
INPUT_VIDEO="${INPUT_VIDEO:-data/lowres/bowing_cif.mp4}"
LEADERBOARD="${LEADERBOARD:-outputs/static_mixed_qat/leaderboard.jsonl}"
FRAMES="${FRAMES:-16}"
BASE_OUT="${BASE_OUT:-outputs/qbasicvsr/bowing_pure_a8w8}"
EVAL_SET="bowing_cif_first16_video_vs_fp16"
REPRO_SCRIPT="scripts/experiments/static_mixed_qat/reproduce_qbasicvsr_bowing_pure_a8w8_static_token_vs_dynamic.sh"

mkdir -p "$BASE_OUT/dynamic_asym" "$BASE_OUT/static_token"

"$PY" - <<'PY'
import json
from pathlib import Path
import torch.nn as nn
from scripts.ptq.fakequant_convert import build_dit

base = Path('outputs/qbasicvsr/bowing_pure_a8w8')
layer_names = [name for name, module in build_dit().named_modules() if isinstance(module, nn.Linear)]
if len(layer_names) != 306:
    raise SystemExit(f"Expected 306 Linear layers, got {len(layer_names)}")

def write_policy(label, qdq_mode, note):
    layers = {
        name: {'mode': 'a8w8', 'activation_qdq_mode': qdq_mode, 'reason': note}
        for name in layer_names
    }
    policy = {
        'schema_version': 'flashvsr.qbasicvsr.pure_a8w8_policy.v2',
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

write_policy('dynamic_asym', 'dynamic_asymmetric', 'pure all-306 DiT Linear A8W8; activation qparams computed online per token')
write_policy('static_token', 'static_token_asymmetric', 'pure all-306 DiT Linear A8W8; frozen per-token qparams captured from dynamic quantized production inference trace')
PY

"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --output "$BASE_OUT/dynamic_asym/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode dynamic_asymmetric \
  --policy "$BASE_OUT/dynamic_asym/policy.json"

"$PY" -u scripts/ptq/qbasicvsr_oracle_trace_calibrate.py \
  --input "$INPUT_VIDEO" \
  --output_cache "$BASE_OUT/static_token/calib_cache.json" \
  --output_video "$BASE_OUT/static_token/dynamic_trace.mp4" \
  --frames "$FRAMES" \
  --checkpoint "$BASE_OUT/dynamic_asym/checkpoint.safetensors" \
  --trace_quantize_mode FakeQuant_A8W8 \
  --mode tiny \
  --scale 4 \
  --precision bf16 \
  --calibration_granularity per_token \
  --tiled_vae --tiled_dit --tile_size 256 --tile_overlap 24 \
  --attention_mode sparse_sage_attention --sparse_ratio 2.0 --kv_ratio 3.0 --local_range 9 \
  > "$BASE_OUT/static_token/calibrate_dynamic_trace.log" 2>&1

"$PY" scripts/ptq/fakequant_convert.py \
  --checkpoint "$FP_CKPT" \
  --calibration_cache "$BASE_OUT/static_token/calib_cache.json" \
  --output "$BASE_OUT/static_token/checkpoint.safetensors" \
  --mode a8w8 \
  --activation_qdq_mode static_token_asymmetric \
  --policy "$BASE_OUT/static_token/policy.json"

"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_bowing16_pure_dynamic_asym_a8w8 \
  --policy "$BASE_OUT/dynamic_asym/policy.json" \
  --checkpoint "$BASE_OUT/dynamic_asym/checkpoint.safetensors" \
  --input_video "$INPUT_VIDEO" --frames "$FRAMES" --eval_set "$EVAL_SET" \
  --clipping_method none --teacher_ft_steps 0 --static_ablation_label pure_dynamic_asym_a8w8 \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD"

FP16_VIDEO="outputs/qbasicvsr/eval/qbasicvsr_bowing16_pure_dynamic_asym_a8w8/fp16.mp4"

"$PY" scripts/ptq/run_qbasicvsr_temporal_eval.py \
  --run_id qbasicvsr_bowing16_pure_static_token_a8w8_dynamic_trace \
  --policy "$BASE_OUT/static_token/policy.json" \
  --checkpoint "$BASE_OUT/static_token/checkpoint.safetensors" \
  --input_video "$INPUT_VIDEO" --frames "$FRAMES" --eval_set "$EVAL_SET" \
  --fp16_video "$FP16_VIDEO" \
  --clipping_method none --teacher_ft_steps 0 --static_ablation_label pure_static_token_a8w8_dynamic_trace \
  --reproduce_script "$REPRO_SCRIPT" --leaderboard "$LEADERBOARD"

"$PY" scripts/experiments/static_mixed_qat/render_leaderboard.py \
  --leaderboard "$LEADERBOARD" \
  --output "${LEADERBOARD%.jsonl}.html"

PREFIX="$(date +%Y%m%d)"
cp "$LEADERBOARD" "/home/user/SynologyDrive/daily/${PREFIX}_flashvsr_static_mixed_leaderboard.jsonl"
cp "${LEADERBOARD%.jsonl}.html" "/home/user/SynologyDrive/daily/${PREFIX}_flashvsr_static_mixed_leaderboard.html"
