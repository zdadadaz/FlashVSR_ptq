#!/usr/bin/env bash
set -euo pipefail

# FP16 teacher inference wrapper for FlashVSR PTQ/QAT experiments.
# Defaults are intentionally the same small bowing CIF first16 baseline used in
# FakeQuant experiments, but every setting can be overridden via env var.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
INPUT="${INPUT:-data/lowres/bowing_cif.mp4}"
OUT_DIR="${OUT_DIR:-outputs/teacher/fp16/${RUN_ID}}"
OUTPUT="${OUTPUT:-${OUT_DIR}/teacher_fp16_first${END_FRAME:-16}.mp4}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}_teacher_fp16_infer.log}"
METADATA="${METADATA:-${OUT_DIR}/teacher_manifest.json}"

MODEL="${MODEL:-FlashVSR-v1.1}"
VAE_MODEL="${VAE_MODEL:-Wan2.1}"
MODE="${MODE:-full}"
PRECISION="${PRECISION:-fp16}"
DEVICE="${DEVICE:-cuda:0}"
ATTENTION_MODE="${ATTENTION_MODE:-sdpa}"
SCALE="${SCALE:-4}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-16}"
SEED="${SEED:-0}"
REF_VIDEO="${REF_VIDEO:-}"
PSNR_JSON="${PSNR_JSON:-${OUT_DIR}/psnr_vs_ref.json}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
export RUN_ID INPUT OUT_DIR OUTPUT LOG_DIR LOG_FILE METADATA MODEL VAE_MODEL MODE PRECISION DEVICE ATTENTION_MODE SCALE START_FRAME END_FRAME SEED REF_VIDEO PSNR_JSON

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

python cli_main.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --vae_model "$VAE_MODEL" \
  --scale "$SCALE" \
  --mode "$MODE" \
  --precision "$PRECISION" \
  --device "$DEVICE" \
  --attention_mode "$ATTENTION_MODE" \
  --quantize_mode None \
  --start_frame "$START_FRAME" \
  --end_frame "$END_FRAME" \
  --seed "$SEED" \
  2>&1 | tee "$LOG_FILE"

python - <<'PY'
import json, os, pathlib, subprocess, sys
out = pathlib.Path(os.environ["OUTPUT"])
manifest = pathlib.Path(os.environ["METADATA"])
manifest.parent.mkdir(parents=True, exist_ok=True)

def stat(path):
    p = pathlib.Path(path)
    return {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0}

data = {
    "schema_version": "flashvsr.teacher.v1",
    "role": "fp16_teacher_output",
    "run_id": os.environ["RUN_ID"],
    "input": os.environ["INPUT"],
    "output": stat(out),
    "log": stat(os.environ["LOG_FILE"]),
    "model": os.environ["MODEL"],
    "vae_model": os.environ["VAE_MODEL"],
    "scope": {
        "teacher_precision": os.environ["PRECISION"],
        "quantize_mode": "None",
        "dit_quantized": False,
        "wan_vae_quantized": False,
    },
    "runtime": {
        "device": os.environ["DEVICE"],
        "mode": os.environ["MODE"],
        "attention_mode": os.environ["ATTENTION_MODE"],
        "scale": int(os.environ["SCALE"]),
        "start_frame": int(os.environ["START_FRAME"]),
        "end_frame": int(os.environ["END_FRAME"]),
        "seed": int(os.environ["SEED"]),
    },
    "artifacts": {
        "teacher_video": str(out),
        "feature_dump_manifest": None,
        "psnr_json": None,
    },
}
manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"[teacher] wrote manifest: {manifest}")
PY

if [[ -n "$REF_VIDEO" ]]; then
  python scripts/compare_video_psnr.py "$REF_VIDEO" "$OUTPUT" --out-json "$PSNR_JSON" \
    2>&1 | tee "${LOG_DIR}/${RUN_ID}_teacher_fp16_psnr.log"
  python - <<'PY'
import json, os, pathlib
manifest = pathlib.Path(os.environ["METADATA"])
data = json.loads(manifest.read_text())
data["artifacts"]["psnr_json"] = os.environ["PSNR_JSON"]
data["quality"] = json.loads(pathlib.Path(os.environ["PSNR_JSON"]).read_text())
manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
fi

echo "[teacher] output=$OUTPUT"
echo "[teacher] manifest=$METADATA"
