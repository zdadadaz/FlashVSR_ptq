# PR9 paper-comparable benchmark update

## PR-9 paper-comparable benchmark slice (2026-06-06)

Added `scripts/ptq/run_lsgquant_pr9_paper_benchmark.py` to run the LSGQuant paper dataset contract: UDM10 / REDS30 / MVSR4x, FP16 baseline vs A4W4 rank-32 QAO, with GT PSNR and PTQ-minus-FP16 deltas.  Wan VAE / decoder remain unquantized.

Artifacts:

- Manifest: `outputs/lsgquant/pr9_paper_benchmark_firstclip16/lsgquant_pr9_paper_benchmark_manifest.json`
- Benchmark runner: `scripts/ptq/run_lsgquant_pr9_paper_benchmark.py`
- Unit tests: `tests/scripts/ptq/test_lsgquant_pr9_paper_benchmark.py`
- A4W4 QAO checkpoint: `outputs/lsgquant/pr9_a4w4_rank32_full_eval/dit_lsgquant_a4w4_rank32_qao.safetensors`

Executed benchmark slice:

- Frames per clip: 16
- Clips: 3 total, `limit_per_dataset=1`
- Metrics: PSNR vs GT and A4W4-QAO minus FP16 PSNR delta

- MVSR4x:
  - FP16 vs GT: 19.8820 dB
  - A4W4-QAO vs GT: 20.2737 dB
  - Delta: +0.3917 dB
- REDS30:
  - FP16 vs GT: 20.4988 dB
  - A4W4-QAO vs GT: 20.2002 dB
  - Delta: -0.2986 dB
- UDM10:
  - FP16 vs GT: 24.3386 dB
  - A4W4-QAO vs GT: 23.9180 dB
  - Delta: -0.4206 dB
- Overall delta: -0.1092 dB

Status: paper-comparable harness and first-clip-per-dataset PSNR slice are complete. Full official paper table still requires running the same harness without `--limit_per_dataset` and wiring optional IQA/VQA metrics beyond PSNR.

## Test verification

- `python -m pytest tests/scripts/ptq/test_lsgquant_pr9_paper_benchmark.py tests/scripts/ptq/test_lsgquant_pr9_a4w4_eval.py tests/scripts/ptq/test_a4_activation_qdq.py -q` → `14 passed in 4.91s`
