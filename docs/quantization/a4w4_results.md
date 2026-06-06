# A4W4 LSGQuant PR-9 Results

Date: 2026-06-06
Scope: WanVideoDiT Linear layers only. Wan VAE / decoder remain unquantized.
Paper reference: LSGQuant, arXiv:2602.03182v1.

## PR-9 implementation contract

PR-9 wires the A4W4 quality-evaluation path rather than claiming deployment speedups.
FakeQuant/LSGQuant checkpoints still perform integer QDQ followed by floating-point
matmul; TensorRT/int-kernel deployment remains a later PR boundary.

Implemented pieces:

- `build_lsgquant_a4w4_fallback_policy(...)` in `src/models/quantization/policy.py`
  - low VOLTS sensitivity → `a4w4`, DRAQ, LSGQuant rank-16
  - mid VOLTS sensitivity → `a4w4`, DRAQ, LSGQuant rank-32, `adaptation=light`
  - high VOLTS sensitivity → `a8w8`, DRAQ, LSGQuant rank-32 fallback
  - optional catastrophic fallback → `fp16_skip` for top-k `mu_var` layers
- `scripts/ptq/lsgquant_policy.py --policy_type a4w4_fallback`
- `scripts/ptq/run_lsgquant_eval_set.py --mode a4w4 --rank --qao_rounds --rotation`
  - emits PR-9 manifest schema `flashvsr.lsgquant.a4w4_eval_manifest.v1`
  - calls `scripts/ptq/lsgquant_convert.py` for QAO conversion
  - records fallback counts, QAO settings, quality-delta placeholders, and deployment boundary
- `scripts/ptq/lsgquant_convert.py` / QAO helpers now accept `a4w4`
  and honor per-layer policy rank/mode fallbacks.

## Example commands

Generate a PR-9 mixed fallback policy:

```bash
python scripts/ptq/lsgquant_policy.py \
  --calibration_cache outputs/calib/lsg_stats.json \
  --out outputs/lsg/policy_a4w4_mixed.json \
  --policy_type a4w4_fallback \
  --delta1 0.001 --delta2 0.075 \
  --fp16_topk 0
```

Dry-run the eval/conversion manifest:

```bash
python scripts/ptq/run_lsgquant_eval_set.py \
  --checkpoint models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors \
  --calibration_cache outputs/calib/lsg_stats.json \
  --policy outputs/lsg/policy_a4w4_mixed.json \
  --mode a4w4 \
  --rank 32 \
  --qao_rounds 4 \
  --out_dir outputs/lsg/a4w4_rank32_eval \
  --dry_run
```

Run actual QAO conversion by removing `--dry_run`. Full video rendering/PSNR should
then be recorded into the emitted manifest/report before making any quality claim.

## Current verification

CPU-only contract tests pass:

```text
32 passed in 8.77s
```

Covered behavior:

- A4 activation fakequant 2D/3D math and CLI exposure.
- A4W4 fallback policy schema and policy-loader validation.
- PR-9 dry-run manifest with QAO command, fallback counts, compression-report boundary,
  and pending quality-delta fields.
- QAO conversion uses per-layer rank and allows high-sensitivity `a8w8` fallback inside
  an `a4w4` default policy.

## Completed PR-9 full-video smoke eval (2026-06-06)

A real A4W4 rank-32 QAO checkpoint was materialized and rendered end-to-end.

Artifacts:

- Eval manifest: `outputs/lsgquant/pr9_a4w4_rank32_full_eval/lsgquant_eval_manifest.json`
- QAO checkpoint: `outputs/lsgquant/pr9_a4w4_rank32_full_eval/dit_lsgquant_a4w4_rank32_qao.safetensors`
- QAO manifest: `outputs/lsgquant/pr9_a4w4_rank32_full_eval/dit_lsgquant_a4w4_rank32_qao.safetensors.manifest.json`
- FP16 render: `outputs/lsgquant/pr9_a4w4_rank32_full_eval/video_render_animal2_full/animal_2_fp16_full.mp4`
- A4W4 render: `outputs/lsgquant/pr9_a4w4_rank32_full_eval/video_render_animal2_full/animal_2_a4w4_rank32_qao_full.mp4`
- PSNR JSON: `outputs/lsgquant/pr9_a4w4_rank32_full_eval/video_render_animal2_full/animal_2_fp16_vs_a4w4_rank32_qao_psnr.json`

Conversion summary:

- QAO layers: 306
- Improved layers: 306
- Mean Frobenius error: 0.744896 → 0.587730
- Mode counts: {'a8w8': 297, 'a4w4': 9}
- Rank counts: {'32': 299, '16': 7}

Full-video render smoke:

- Input: `data/lowres/animal_2.mp4` from the main FlashVSR worktree
- Frames: 150 / 150
- Settings: scale=4, mode=tiny, bf16, SDPA, resize_factor=0.25, frame_chunk_size=10, no color fix
- Candidate integrity: decoded successfully as uint8 video; no NaN/Inf-corruption signal.
- PSNR vs same-settings FP16 baseline: avg 27.1545 dB, min 26.3113 dB, max 34.4271 dB.

Status: PR-9 success criteria are now completed for an end-to-end full-video smoke run.
This is still not a paper-comparable UDM10/REDS30/MVSR4x benchmark.

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
