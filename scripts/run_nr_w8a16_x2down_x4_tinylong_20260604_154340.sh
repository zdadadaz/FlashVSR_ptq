#!/usr/bin/env bash
set -euo pipefail

REPO=/home/user/apps/FlashVSRptq/FlashVSR_Integrated
SRC_DIR=/home/user/data/nr
RUN_ID=20260604_154340_flashvsr_w8a16_nr_x2down_x4_tinylong_first3s
OUT_ROOT=/home/user/SynologyDrive/daily/${RUN_ID}
CKPT=models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors

cd "$REPO"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/trimmed_3s" "$OUT_ROOT/downsample_x2_bicubic_antialias" "$OUT_ROOT/flashvsr_x4_w8a16_tinylong" "$OUT_ROOT/logs" "$OUT_ROOT/tmp"
exec > >(tee -a "$OUT_ROOT/logs/run.log") 2>&1

echo "RUN_ID=$RUN_ID"
echo "REPO=$REPO"
echo "SRC_DIR=$SRC_DIR"
echo "OUT_ROOT=$OUT_ROOT"
date --iso-8601=seconds

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint missing: $CKPT" >&2
  exit 2
fi

source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -d '' VIDEOS < <(python - <<'PY'
from pathlib import Path
exts={'.mp4','.mov','.mkv','.avi','.webm','.m4v','.m2v','.tp','.ts','.mpeg','.mpg'}
for p in sorted(Path('/home/user/data/nr').rglob('*')):
    if p.is_file() and p.suffix.lower() in exts:
        print(str(p)+'\0', end='')
PY
)

if (( ${#VIDEOS[@]} == 0 )); then
  echo "ERROR: no videos found under $SRC_DIR" >&2
  exit 3
fi

echo -e "source\ttrimmed\tdownsample_x2\tflashvsr_x4\tstatus" > "$OUT_ROOT/manifest.tsv"

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
  CLIP="$OUT_ROOT/trimmed_3s/${SAFE}_first3s.mp4"
  LR="$OUT_ROOT/downsample_x2_bicubic_antialias/${SAFE}_first3s_x2down_bicubic_antialias.mp4"
  OUT="$OUT_ROOT/flashvsr_x4_w8a16_tinylong/${SAFE}_first3s_x2down_flashvsr_x4_w8a16_tinylong.mp4"
  FRAMES_DIR="$OUT_ROOT/tmp/${SAFE}_frames"
  mkdir -p "$FRAMES_DIR"

  echo "===== Processing $SRC ====="
  echo "[1/3] trim first 3 seconds -> $CLIP"
  ffmpeg -y -hide_banner -loglevel warning -i "$SRC" -t 3 \
    -map 0:v:0 -an -vf "setsar=1" \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p "$CLIP"

  echo "[2/3] PyTorch bicubic downsample x2 with antialias=True -> $LR"
  python - <<PY
from pathlib import Path
import json, subprocess
import cv2
import torch
import torch.nn.functional as F
clip = Path(r"$CLIP")
frames_dir = Path(r"$FRAMES_DIR")
lr = Path(r"$LR")
frames_dir.mkdir(parents=True, exist_ok=True)
for old in frames_dir.glob('*.png'):
    old.unlink()
probe = subprocess.check_output([
    'ffprobe','-v','error','-select_streams','v:0',
    '-show_entries','stream=avg_frame_rate,r_frame_rate,width,height',
    '-of','json',str(clip)
], text=True)
info = json.loads(probe)['streams'][0]
fps_expr = info.get('avg_frame_rate') or info.get('r_frame_rate') or '30/1'
cap = cv2.VideoCapture(str(clip))
if not cap.isOpened():
    raise RuntimeError(f'cannot open {clip}')
idx = 0
orig_w = orig_h = None
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    h, w = bgr.shape[:2]
    orig_h, orig_w = h, w
    new_h = max(1, h // 2)
    new_w = max(1, w // 2)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
    y = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)
    y = y.clamp(0,1).mul(255).round().byte().squeeze(0).permute(1,2,0).cpu().numpy()
    cv2.imwrite(str(frames_dir / f'{idx:06d}.png'), cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
    idx += 1
cap.release()
if idx == 0:
    raise RuntimeError(f'no frames decoded from {clip}')
print(f'downsampled {idx} frames: {orig_w}x{orig_h} -> {orig_w//2}x{orig_h//2}, fps={fps_expr}, antialias=True')
subprocess.check_call([
    'ffmpeg','-y','-hide_banner','-loglevel','warning',
    '-framerate', fps_expr,
    '-i', str(frames_dir / '%06d.png'),
    '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p', str(lr)
])
PY

  echo "[3/3] FlashVSR W8A16 mode=tiny-long scale=4 -> $OUT"
  python cli_main.py \
    --input "$LR" \
    --output "$OUT" \
    --model FlashVSR-v1.1 \
    --scale 4 \
    --quantize_mode W8A16 \
    --ckpt_path "$CKPT" \
    --device cuda:0 \
    --precision fp16 \
    --mode tiny-long \
    --vae_model Wan2.1 \
    --tiled_vae \
    --tiled_dit \
    --tile_size 256 \
    --tile_overlap 32 \
    --frame_chunk_size 26 \
    --codec libx264 \
    --crf 18 \
    --enable_debug 2>&1 | tee "$OUT_ROOT/logs/${SAFE}_flashvsr.log"

  # Preserve source/LR fps if FlashVSR writer normalizes FPS internally.
  FPS_EXPR=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$LR")
  OUT_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$OUT")
  if [[ "$OUT_FPS" != "$FPS_EXPR" ]]; then
    echo "Retiming $OUT from $OUT_FPS to $FPS_EXPR"
    TMP="${OUT%.mp4}.fpsfix.mp4"
    ffmpeg -y -hide_banner -loglevel warning -r "$FPS_EXPR" -i "$OUT" \
      -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p "$TMP"
    mv "$TMP" "$OUT"
  fi

  probe_json "$SRC" "$OUT_ROOT/logs/${SAFE}_source_probe.json" || true
  probe_json "$CLIP" "$OUT_ROOT/logs/${SAFE}_trimmed_probe.json" || true
  probe_json "$LR" "$OUT_ROOT/logs/${SAFE}_downsample_probe.json" || true
  probe_json "$OUT" "$OUT_ROOT/logs/${SAFE}_output_probe.json"

  echo -e "$SRC\t$CLIP\t$LR\t$OUT\tok" >> "$OUT_ROOT/manifest.tsv"
  rm -rf "$FRAMES_DIR"
done

python - <<'PY'
from pathlib import Path
import json
root = Path('/home/user/SynologyDrive/daily/20260604_154340_flashvsr_w8a16_nr_x2down_x4_tinylong_first3s')
lines = [
    '# FlashVSR W8A16 NR first-3s x2-down x4-up report',
    '',
    '- Source: `/home/user/data/nr`',
    '- Scope: first 3 seconds of each video',
    '- Preprocess: PyTorch `F.interpolate(mode="bicubic", antialias=True)` downsample x2, saved as mp4',
    '- Inference: FlashVSR-v1.1 `W8A16`, `mode=tiny-long`, `scale=4` (fallback because `mode=full` OOM at x2-down/4K output)',
    '- Checkpoint: `models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors`',
    '- VRAM safety: CLI ran with `--tiled_vae --tiled_dit --tile_size 256 --tile_overlap 32 --frame_chunk_size 26`',
    '',
    '## Artifacts',
    f'- Root: `{root}`',
    f'- Manifest: `{root / "manifest.tsv"}`',
    f'- Downsample videos: `{root / "downsample_x2_bicubic_antialias"}`',
    f'- FlashVSR outputs: `{root / "flashvsr_x4_w8a16_tinylong"}`',
    f'- Logs/probes: `{root / "logs"}`',
    '',
    '## Outputs',
]
manifest = root / 'manifest.tsv'
if manifest.exists():
    for row in manifest.read_text().splitlines()[1:]:
        src, clip, lr, out, status = row.split('\t')
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
