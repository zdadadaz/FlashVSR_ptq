#!/usr/bin/env bash
set -euo pipefail

REPO=/home/user/apps/FlashVSRptq/FlashVSR_Integrated
SRC_DIR=/home/user/data/nr
RUN_ID=20260604_152629_flashvsr_w8a16_nr_x4down_x2full
OUT_ROOT=/home/user/SynologyDrive/daily/${RUN_ID}
CKPT=models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors

cd "$REPO"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/trimmed_3s" "$OUT_ROOT/downsample_x4_bicubic_antialias" "$OUT_ROOT/flashvsr_x2_w8a16_full" "$OUT_ROOT/logs" "$OUT_ROOT/tmp"

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

# Discover broad video extensions recursively, deterministic order.
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

echo -e "source\ttrimmed\tdownsample_x4\tflashvsr_x2\tstatus" > "$OUT_ROOT/manifest.tsv"

# Helper: write ffprobe JSON for an artifact.
probe_json() {
  local f="$1" out="$2"
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration -show_entries format=duration,size -of json "$f" > "$out"
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
  LR="$OUT_ROOT/downsample_x4_bicubic_antialias/${SAFE}_first3s_x4down_bicubic_antialias.mp4"
  OUT="$OUT_ROOT/flashvsr_x2_w8a16_full/${SAFE}_first3s_x4down_flashvsr_x2_w8a16_full.mp4"
  FRAMES_DIR="$OUT_ROOT/tmp/${SAFE}_frames"
  mkdir -p "$FRAMES_DIR"

  echo "===== Processing $SRC ====="
  echo "[1/3] trim first 3 seconds -> $CLIP"
  ffmpeg -y -hide_banner -loglevel warning -i "$SRC" -t 3 \
    -map 0:v:0 -an -vf "setsar=1" \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p "$CLIP"

  echo "[2/3] PyTorch bicubic downsample x4 with antialias=True -> $LR"
  python - <<PY
from pathlib import Path
import json, subprocess, math, shutil
import cv2
import torch
import torch.nn.functional as F

clip = Path(r"$CLIP")
frames_dir = Path(r"$FRAMES_DIR")
lr = Path(r"$LR")
frames_dir.mkdir(parents=True, exist_ok=True)
for old in frames_dir.glob('*.png'):
    old.unlink()

# Determine FPS as float and rational string for ffmpeg.
probe = subprocess.check_output([
    'ffprobe','-v','error','-select_streams','v:0',
    '-show_entries','stream=avg_frame_rate,r_frame_rate,width,height',
    '-of','json',str(clip)
], text=True)
info = json.loads(probe)['streams'][0]
fps_expr = info.get('avg_frame_rate') or info.get('r_frame_rate') or '30/1'
try:
    num, den = fps_expr.split('/')
    fps_float = float(num) / float(den)
except Exception:
    fps_float = 30.0

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
    new_h = max(1, h // 4)
    new_w = max(1, w // 4)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
    y = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)
    y = y.clamp(0,1).mul(255).round().byte().squeeze(0).permute(1,2,0).cpu().numpy()
    out_bgr = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(frames_dir / f'{idx:06d}.png'), out_bgr)
    idx += 1
cap.release()
if idx == 0:
    raise RuntimeError(f'no frames decoded from {clip}')
print(f'downsampled {idx} frames: {orig_w}x{orig_h} -> {orig_w//4}x{orig_h//4}, fps={fps_expr}, antialias=True')
subprocess.check_call([
    'ffmpeg','-y','-hide_banner','-loglevel','warning',
    '-framerate', fps_expr,
    '-i', str(frames_dir / '%06d.png'),
    '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p', str(lr)
])
PY

  echo "[3/3] FlashVSR W8A16 mode=full scale=2 -> $OUT"
  python cli_main.py \
    --input "$LR" \
    --output "$OUT" \
    --model FlashVSR-v1.1 \
    --scale 2 \
    --quantize_mode W8A16 \
    --ckpt_path "$CKPT" \
    --device cuda:0 \
    --precision fp16 \
    --mode full \
    --vae_model Wan2.1 \
    --codec libx264 \
    --crf 18 \
    --enable_debug 2>&1 | tee "$OUT_ROOT/logs/${SAFE}_flashvsr.log"

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
root = Path('/home/user/SynologyDrive/daily/20260604_152629_flashvsr_w8a16_nr_x4down_x2full')
lines = [
    '# FlashVSR W8A16 NR batch inference report',
    '',
    '- Source: `/home/user/data/nr`',
    '- Scope: first 3 seconds of each video',
    '- Preprocess: PyTorch `F.interpolate(mode="bicubic", antialias=True)` downsample x4, saved as mp4',
    '- Inference: FlashVSR-v1.1 `W8A16`, `mode=full`, `scale=2`',
    '- Checkpoint: `models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors`',
    '',
    '## Artifacts',
    f'- Root: `{root}`',
    f'- Manifest: `{root / "manifest.tsv"}`',
    f'- Downsample videos: `{root / "downsample_x4_bicubic_antialias"}`',
    f'- FlashVSR outputs: `{root / "flashvsr_x2_w8a16_full"}`',
    f'- Logs/probes: `{root / "logs"}`',
    '',
    '## Outputs',
]
manifest = root / 'manifest.tsv'
if manifest.exists():
    for row in manifest.read_text().splitlines()[1:]:
        src, clip, lr, out, status = row.split('\t')
        lines.append(f'- `{Path(src).name}` → `{out}` ({status})')
(root / 'report.md').write_text('\n'.join(lines) + '\n')
print(root / 'report.md')
PY

echo "DONE: $OUT_ROOT"
date --iso-8601=seconds
