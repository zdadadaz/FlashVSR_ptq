#!/usr/bin/env bash
set -euo pipefail

REPO=/home/user/apps/FlashVSRptq/FlashVSR_Integrated
SRC_DIR=/home/user/data/nr/mobile
RUN_ID=20260604_160844_flashvsr_w8a16_mobile_x4down_x4full
OUT_ROOT=/home/user/SynologyDrive/daily/${RUN_ID}
CKPT=models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors

cd "$REPO"
mkdir -p "$OUT_ROOT/flashvsr_x4_w8a16_full_resize025" "$OUT_ROOT/logs"
exec > >(tee -a "$OUT_ROOT/logs/run.log") 2>&1

echo "RUN_ID=$RUN_ID"
echo "REPO=$REPO"
echo "SRC_DIR=$SRC_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "CKPT=$CKPT"
date --iso-8601=seconds

if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERROR: source directory missing: $SRC_DIR" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint missing: $CKPT" >&2
  exit 3
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: repo venv missing: $REPO/.venv" >&2
  exit 4
fi

source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -d '' VIDEOS < <(python - <<'PY'
from pathlib import Path
exts={'.mp4','.mov','.mkv','.avi','.webm','.m4v','.m2v','.tp','.ts','.mpeg','.mpg'}
for p in sorted(Path('/home/user/data/nr/mobile').rglob('*')):
    if p.is_file() and p.suffix.lower() in exts:
        print(str(p)+'\0', end='')
PY
)

if (( ${#VIDEOS[@]} == 0 )); then
  echo "ERROR: no videos found under $SRC_DIR" >&2
  exit 5
fi

echo -e "source\tflashvsr_x4\tstatus" > "$OUT_ROOT/manifest.tsv"

probe_json() {
  local f="$1" out="$2"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration \
    -show_entries format=duration,size -of json "$f" > "$out"
}

for SRC in "${VIDEOS[@]}"; do
  BASE=$(basename "$SRC")
  STEM="${BASE%.*}"
  SAFE=$(python - <<PY
import re
print(re.sub(r'[^A-Za-z0-9._-]+','_', '$STEM'))
PY
)
  OUT="$OUT_ROOT/flashvsr_x4_w8a16_full_resize025/${SAFE}_resize025_flashvsr_x4_w8a16_full.mp4"

  echo "===== Processing $SRC ====="
  probe_json "$SRC" "$OUT_ROOT/logs/${SAFE}_source_probe.json" || true

  echo "FlashVSR W8A16 mode=full scale=4, internal bicubic antialias downsample x4 via --resize_factor 0.25 -> $OUT"
  rm -f "$OUT"
  python cli_main.py \
    --input "$SRC" \
    --output "$OUT" \
    --model FlashVSR-v1.1 \
    --scale 4 \
    --resize_factor 0.25 \
    --quantize_mode W8A16 \
    --ckpt_path "$CKPT" \
    --device cuda:0 \
    --precision fp16 \
    --mode full \
    --vae_model Wan2.1 \
    --tiled_vae \
    --tiled_dit \
    --tile_size 512 \
    --tile_overlap 32 \
    --frame_chunk_size 16 \
    --codec libx264 \
    --crf 18 \
    --enable_debug 2>&1 | tee "$OUT_ROOT/logs/${SAFE}_flashvsr.log"

  probe_json "$OUT" "$OUT_ROOT/logs/${SAFE}_output_probe.json"
  echo -e "$SRC\t$OUT\tok" >> "$OUT_ROOT/manifest.tsv"
done

python - <<'PY'
from pathlib import Path
import json
root = Path('/home/user/SynologyDrive/daily/20260604_160844_flashvsr_w8a16_mobile_x4down_x4full')
lines = [
    '# FlashVSR W8A16 mobile x4-down x4-up report',
    '',
    '- Source: `/home/user/data/nr/mobile`',
    '- Preprocess: cli_main.py `--resize_factor 0.25`; implementation uses PyTorch `F.interpolate(mode="bicubic", antialias=True)` before FlashVSR',
    '- Inference: FlashVSR-v1.1 `W8A16`, `mode=full`, `scale=4`',
    '- Checkpoint: `models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors`',
    '- VRAM safety: `--tiled_vae --tiled_dit --tile_size 512 --tile_overlap 32 --frame_chunk_size 16`',
    '',
    '## Artifacts',
    f'- Root: `{root}`',
    f'- Manifest: `{root / "manifest.tsv"}`',
    f'- FlashVSR outputs: `{root / "flashvsr_x4_w8a16_full_resize025"}`',
    f'- Logs/probes: `{root / "logs"}`',
    '',
    '## Outputs',
]
manifest = root / 'manifest.tsv'
if manifest.exists():
    for row in manifest.read_text().splitlines()[1:]:
        src, out, status = row.split('\t')
        lines.append(f'- `{Path(src).name}` → `{out}` ({status})')
lines += ['', '## Verification']
for probe in sorted((root/'logs').glob('*_output_probe.json')):
    stem = probe.name.replace('_output_probe.json','')
    data = json.loads(probe.read_text())
    st = data['streams'][0]
    lines.append(f'- `{stem}` output: {st.get("width")}x{st.get("height")}, frames={st.get("nb_frames")}, duration={st.get("duration")}s, fps={st.get("avg_frame_rate")}')
(root / 'report.md').write_text('\n'.join(lines) + '\n')
print(root / 'report.md')
PY

echo "DONE: $OUT_ROOT"
date --iso-8601=seconds
