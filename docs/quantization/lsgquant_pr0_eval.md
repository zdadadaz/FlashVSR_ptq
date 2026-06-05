# LSGQuant PR-0 Evaluation Harness

PR-0 records the experiment setting from LSGQuant §4.1 and produces a reproducible FlashVSR FP16/PTQ evaluation manifest. It does **not** add a new quantizer yet.

## Paper setting encoded

- Calibration dataset: HQ-VSR
- Calibration sample count: 50 randomly sampled videos
- Calibration procedure: run full inference with the FP DiT model
- Evaluation datasets:
  - UDM10: synthetic
  - REDS30: synthetic
  - MVSR4x: real-world
- Metrics contract:
  - Reference IQA: DISTS, PSNR, SSIM, LPIPS
  - No-reference IQA: MANIQA, CLIP-IQA, MUSIQ
  - VQA: Ewarp*, DOVER
- VOLTS constants: δ1=0.001, δ2=0.075; frozen=1 iteration, light=30 iterations, full=until convergence
- Weight quantization setting: static asymmetric channel-wise quantizer
- SVD rank for future QAO path: r=32
- Scope: WanVideoDiT Linear layers only; Wan VAE remains unquantized

## Script

```bash
.venv/bin/python scripts/ptq/lsgquant_pr0_eval.py \
  --calibration_dataset /path/to/HQ-VSR \
  --eval_dataset UDM10=/path/to/UDM10 \
  --eval_dataset REDS30=/path/to/REDS30 \
  --eval_dataset MVSR4x=/path/to/MVSR4x \
  --out_dir outputs/lsgquant/pr0_eval \
  --fp_checkpoint /path/to/fp_dit.safetensors \
  --quant_checkpoint /path/to/dit_a8w8.safetensors \
  --frames 16 \
  --seed 0
```

Default mode is dry-run/manifest-only. Add `--execute` to actually run the generated FlashVSR CLI render commands.

The output manifest is:

```text
outputs/lsgquant/pr0_eval/lsgquant_pr0_eval_manifest.json
```

## Manifest contents

The manifest includes:

- `experiment_settings`: exact §4.1 constants and metric plan
- `calibration`: deterministic 50-video HQ-VSR sample
- `evaluation`: discovered videos for UDM10/REDS30/MVSR4x
- `runs`: FP16 and optional PTQ FlashVSR CLI commands per clip
- `metric_plan`: implemented-now PSNR plus planned paper metrics

## Tests

```bash
.venv/bin/python -m pytest tests/scripts/ptq/test_lsgquant_pr0_eval.py -v
```
