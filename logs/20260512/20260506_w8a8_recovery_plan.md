# FlashVSR W8A8 SmoothQuant Recovery Plan: Refined

## Status Report (2026-05-06)
- **Current W8A16 PSNR:** 35.57 dB (target >30 dB — ACHIEVED)
- **Current W8A8 PSNR:** 8.83 dB (Catastrophic Failure)
- **Root Cause Identified:** Calibration forward passes fail due to Cross-Attn KV not initialized

## Critique of Previous Plan (20260505_PTQ_Recovery_Plan.md)

The previous plan hypothesized:
1. **"Calibration Data Gap: Current calibration uses random noise"** — INCORRECT. Calibration uses real videos from SPMCS, UDM10, etc.
2. **"Timestep Sensitivity"** — Unproven. Diffusion timestep sampling not confirmed as issue.
3. **"Outlier Mismanagement: Fixed α=0.5 might be suboptimal"** — May be valid, but not the primary failure cause.
4. **"Softmax/LayerNorm Noise"** — May be valid, but masked by calibration failure.

**The actual root cause** was calibration forward pass failure: all 3 calibration videos failed with `"Cross-Attn KV not initialized"` before any meaningful activation statistics could be collected.

## Root Cause Analysis (Verified)

### Primary: KV Cache Not Initialized
- `collect_activation_stats()` in `smoothquant.py` calls `flashvsr()` which requires KV cache to be initialized
- `init_pipeline()` calls `pipe.init_cross_kv()` AFTER quantization conversion (line 934)
- Without KV cache, DiT cross-attention layers produce undefined/unstable activations
- Result: ObserverLinear collects garbage activation amax → broken SmoothQuant scales → 8.83 dB crash

### Secondary: Simplified Forward Pass Fallback Missing
- When `flashvsr()` fails, the fallback in `collect_activation_stats()` silently continues with empty `act_stats`
- No error propagation, no retry with simpler forward pass
- The code at `smoothquant.py:129-172` silently swallows exceptions

## Strategic Action Items

### Phase 1: Fix Calibration Infrastructure (CRITICAL)
- [ ] **Task 1.1:** Initialize KV cache before running calibration forward pass in `collect_activation_stats()`
- [ ] **Task 1.2:** Add fallback dummy forward pass (random latent + context) when real inference fails
- [ ] **Task 1.3:** Verify `act_stats` is non-empty before proceeding with conversion
- [ ] **Task 1.4:** Run sensitivity analysis to identify which specific DiT blocks break when miscalibrated

### Phase 2: Alpha Optimization (High Priority)
- [ ] **Task 2.1:** Per-layer alpha search in range [0.4, 0.95] for attention vs FFN blocks
- [ ] **Task 2.2:** Use different alpha for Attention.qkv, Attention.proj, FFN.fc1, FFN.fc2
- [ ] **Task 2.3:** Add bias compensation (mean-error correction) post-quantization

### Phase 3: Architectural Guardrails (Medium Priority)
- [ ] **Task 3.1:** Keep Softmax, LayerNorm, first/last DiT layers in FP16/A16
- [ ] **Task 3.2:** Per-channel weight quantization to handle weight outliers
- [ ] **Task 3.3:** Verify VRAM usage stays <12GB for 2x upscale

## Updated PR Roadmap

| PR ID | Name | Objective | Status |
| :--- | :--- | :--- | :--- |
| **PR #10** | **Diagnostic Toolset** | DOVE Loader + Sensitivity Analysis script | Pending |
| **PR #11** | **Calibration Fix** | KV init + fallback dummy pass | **IN PROGRESS** |
| **PR #12** | **SmoothQuant Optimization** | Layer-wise α + Bias Correction | Pending |
| **PR #13** | **Production W8A8** | Final integration with ComfyUI nodes and CLI | Pending |

## Technical Notes

### KV Cache Initialization Fix
```python
# In collect_activation_stats(), initialize KV cache BEFORE running inference:
pipe.init_cross_kv(prompt_path=prompt_path)  # Must happen before flashvsr()
```

### Fallback Dummy Forward Pass
When real inference fails, use a simple tensor pass that doesn't require KV cache:
```python
# Dummy forward pass for calibration
x = torch.randn(1, 1, 16, 24, 24, device='cuda', dtype=torch.bfloat16)
t = torch.randint(0, 1000, (1,), device='cuda')
context = torch.randn(1, 10, 4096, device='cuda').to(x.dtype)
# Run model forward directly without flashvsr wrapper
```

### Alpha Per-Layer Strategy
Based on SmoothQuant paper, different layer types benefit from different alpha:
- **Attention.qkv:** α=0.7 (activations have higher variance)
- **Attention.proj:** α=0.5
- **FFN.fc1:** α=0.4 (weight-centric)
- **FFN.fc2:** α=0.6 (activation-centric)

## Success Criteria
- W8A8 PSNR Recovery: >28 dB within 48 hours
- Calibration must complete successfully on all 3+ videos without KV errors
- Stability: No visual artifacts in generated video upscales
- Efficiency: Maintain <12GB VRAM usage for 2x upscale
