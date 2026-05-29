#!/usr/bin/env python3
"""Run FlashVSR FakeQuant extra-scope sensitivity and PSNR report."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAILY = Path('/home/user/SynologyDrive/daily')
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTDIR = DAILY / f'{STAMP}_flashvsr_fakequant_extra_scopes'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Use the project venv explicitly. Background shells may resolve `python` to base conda.
PY = str(ROOT / '.venv/bin/python') if (ROOT / '.venv/bin/python').exists() else sys.executable
INPUT = ROOT / 'data/lowres/bowing_cif.mp4'
CKPT = ROOT / 'models/FlashVSR-v1.1/fakequant_a8w8.safetensors'
COMMON = [
    PY, 'cli_main.py',
    '--input', str(INPUT),
    '--scale', '4',
    '--mode', 'full',
    '--vae_model', 'Wan2.1',
    '--precision', 'bf16',
    '--device', 'cuda:0',
    '--end_frame', '16',
    '--frame_chunk_size', '16',
    '--crf', '18',
]
RUNS = [
    {'name': 'fp16_baseline', 'args': ['--quantize_mode', 'None']},
    {'name': 'dit_linear_a8w8', 'args': ['--quantize_mode', 'FakeQuant_A8W8', '--ckpt_path', str(CKPT)]},
    {'name': 'dit_plus_lq_proj_in', 'args': ['--quantize_mode', 'FakeQuant_A8W8', '--ckpt_path', str(CKPT), '--fakequant_extra_scopes', 'lq_proj_in']},
    {'name': 'dit_plus_tcdecoder', 'args': ['--quantize_mode', 'FakeQuant_A8W8', '--ckpt_path', str(CKPT), '--fakequant_extra_scopes', 'tcdecoder']},
    {'name': 'dit_plus_wan_vae', 'args': ['--quantize_mode', 'FakeQuant_A8W8', '--ckpt_path', str(CKPT), '--fakequant_extra_scopes', 'wan_vae']},
    {'name': 'dit_plus_dit_conv3d', 'args': ['--quantize_mode', 'FakeQuant_A8W8', '--ckpt_path', str(CKPT), '--fakequant_extra_scopes', 'dit_conv3d']},
    {'name': 'dit_plus_all_extra', 'args': ['--quantize_mode', 'FakeQuant_A8W8', '--ckpt_path', str(CKPT), '--fakequant_extra_scopes', 'all']},
]

def run(cmd, log_path):
    print('$ ' + ' '.join(map(str, cmd)), flush=True)
    with open(log_path, 'w') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}); see {log_path}')

results = []
for r in RUNS:
    out = OUTDIR / f"{r['name']}.mp4"
    log = OUTDIR / f"{r['name']}.log"
    cmd = COMMON + ['--output', str(out)] + r['args']
    run(cmd, log)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f'expected output video missing/empty after {r["name"]}; see {log}')
    r['output'] = str(out)
    r['log'] = str(log)

ref = OUTDIR / 'fp16_baseline.mp4'
for r in RUNS[1:]:
    psnr_json = OUTDIR / f"psnr_{r['name']}_vs_fp16.json"
    cmd = [PY, 'scripts/compare_video_psnr.py', str(ref), r['output'], '--out-json', str(psnr_json)]
    run(cmd, OUTDIR / f"psnr_{r['name']}.log")
    data = json.loads(psnr_json.read_text())
    r['psnr_json'] = str(psnr_json)
    r['psnr_avg_db'] = data['psnr_avg_db']
    r['psnr_min_db'] = data['psnr_min_db']
    r['frames'] = data['frames']
    results.append(r)

summary = {
    'timestamp': STAMP,
    'input': str(INPUT),
    'frames': 16,
    'reference': str(ref),
    'runs': results,
}
(OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2))

base = next(x for x in results if x['name'] == 'dit_linear_a8w8')
lines = []
lines.append(f"# FlashVSR FakeQuant A8W8 extra-scope sensitivity report ({STAMP})\n")
lines.append(f"- Input: `{INPUT}` first 16 frames, scale=4, mode=full, vae=Wan2.1, precision=bf16")
lines.append(f"- FP16 reference: `{ref}`")
lines.append(f"- Base FakeQuant checkpoint: `{CKPT}`")
lines.append(f"- Output dir: `{OUTDIR}`\n")
lines.append("## PSNR vs FP16\n")
for r in results:
    delta = r['psnr_avg_db'] - base['psnr_avg_db']
    lines.append(f"- {r['name']}: avg={r['psnr_avg_db']:.4f} dB, min={r['psnr_min_db']:.4f} dB, delta_vs_dit_linear={delta:+.4f} dB, json=`{r['psnr_json']}`")
lines.append("\n## Interpretation\n")
worst = min(results, key=lambda x: x['psnr_avg_db'])
lines.append(f"- Largest drop in this run: `{worst['name']}` ({worst['psnr_avg_db']:.4f} dB avg vs FP16).")
lines.append("- `dit_linear_a8w8` isolates existing DiT Linear FakeQuant. Each `dit_plus_*` run adds one extra quantized component on top of that baseline.")
lines.append("- Extra op quantization uses true int8 QDQ for activation and weight, then float32 conv/linear compute. It is a quality sensitivity path, not an optimized int8 kernel path.")
lines.append("\n## Quantized scopes implemented\n")
lines.append("- `wan_vae`: Wan VAE decoder-side remaining Linear/Conv2d/Conv3d ops after encoder/conv1 are removed in this pipeline.")
lines.append("- `tcdecoder`: TCDecoder Linear/Conv2d/Conv3d ops.")
lines.append("- `lq_proj_in`: LQ projection Linear/Conv2d/Conv3d ops.")
lines.append("- `dit_conv3d`: DiT Conv3d ops such as patch embedding, in addition to the FakeQuant checkpoint's DiT Linear ops.")
report = OUTDIR / f'{STAMP}_flashvsr_fakequant_extra_scopes_report.md'
report.write_text('\n'.join(lines))
print(json.dumps({'outdir': str(OUTDIR), 'report': str(report), 'summary': str(OUTDIR / 'summary.json')}, indent=2))
