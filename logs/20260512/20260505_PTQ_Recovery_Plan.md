# FlashVSR PTQ Recovery Plan: W8A8 PSNR Restoration

## Status Report (2026-05-05)
- **Current Metric:** 9.07 dB PSNR (Catastrophic Failure)
- **Baseline Target:** >30 dB PSNR
- **Hardware Target:** W8A8 (SmoothQuant)
- **Calibration Status:** DOVE Dataset (HQ-VSR, UDM10) successfully downloaded.

## Root Cause Analysis (Hypotheses)
1. **Calibration Data Gap:** Current calibration uses random noise, which fails to capture DiT activation distributions.
2. **Timestep Sensitivity:** Diffusion models have vastly different activation scales at different timesteps ($t=0$ vs $t=T$).
3. **Outlier Mismanagement:** Fixed $\alpha=0.5$ in SmoothQuant might be suboptimal for Attention layers.
4. **Softmax/LayerNorm Noise:** Quantizing non-linear layers in INT8 often causes massive instability.

## Strategic Action Items

### Phase 1: Diagnostic & Baseline (High Priority)
- [ ] **Task 1.1: DOVE Data Loader:** Update `scripts/ptq_calibrate.py` to sample real frames from `datasets/train/HQ-VSR`.
- [ ] **Task 1.2: FP16 Baseline:** Establish a gold-standard PSNR using the `UDM10` test set in FP16.
- [ ] **Task 1.3: Per-Layer Sensitivity Analysis:** Create a script to quantize one layer at a time to identify "Killer Layers" that cause the 9dB crash.

### Phase 2: Refined SmoothQuant (W8A8+)
- [ ] **Task 2.1: Timestep-Aware Calibration:** Collect activation statistics for 4-8 discrete timestep buckets.
- [ ] **Task 2.2: Layer-wise Alpha Search:** Implement a search for the best migration strength $\alpha \in [0.4, 0.95]$ per block.
- [ ] **Task 2.3: Bias Compensation:** Implement mean-error correction for Linear layer biases post-quantization.

### Phase 3: Architectural Guardrails
- [ ] **Task 3.1: Selective Precision:** Force Softmax, LayerNorm, and the first/last DiT layers to remain in FP16/A16.
- [ ] **Task 3.2: Per-Channel Weight Quantization:** Ensure weights use per-channel scaling to handle weight outliers.

## Updated PR Roadmap

| PR ID | Name | Objective |
| :--- | :--- | :--- |
| **PR #10** | **Diagnostic Toolset** | DOVE Loader + Sensitivity Analysis script. |
| **PR #11** | **Timestep Calibration** | Multi-timestep scale collection + Averaging logic. |
| **PR #12** | **SmoothQuant Optimization** | Layer-wise $\alpha$ + Bias Correction. |
| **PR #13** | **Production W8A8** | Final integration with ComfyUI nodes and CLI. |

## Success Criteria
- PSNR Recovery: >28 dB within 48 hours.
- Stability: No visual artifacts in generated video upscales.
- Efficiency: Maintain <12GB VRAM usage for 2x upscale.
