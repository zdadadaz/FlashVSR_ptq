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

## Quality status

No paper-comparable A4W4 video-quality numbers are claimed in this PR. The manifest
contains `quality_delta.status = pending_eval` until full FlashVSR renders complete
without NaN/Inf and PSNR/SSIM/LPIPS/temporal metrics are filled.
