# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashVSR Integrated is a ComfyUI custom node + standalone CLI for high-performance video super resolution using diffusion models. It supports 5 VAE architectures and intelligent VRAM management for 8GB-24GB+ GPUs.

## Common Commands

```bash
# Run tests (mock-based, no GPU required)
python -m pytest test_mock.py -v

# Run via CLI
python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

# Low VRAM CLI example (8GB GPUs)
python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 \
    --vae_model LightVAE_W2.1 --tiled_vae --tiled_dit \
    --frame_chunk_size 20 --resize_factor 0.5

# Install block sparse attention (requires CUDA)
python install_block_sparse_attention.py

# PTQ calibration
python scripts/ptq_calibrate.py
```

## Architecture

### Entry Points
- `nodes.py` - ComfyUI node definitions (FlashVSRNodeInitPipe, FlashVSRNode, FlashVSRNodeAdv)
- `cli_main.py` - Standalone CLI mirroring all node parameters

### Core Source (`src/`)
- `pipelines/` - Three pipeline modes with different VRAM/quality tradeoffs:
  - `flashvsr_full.py` - Maximum quality, highest VRAM
  - `flashvsr_tiny.py` - Standard fast mode
  - `flashvsr_tiny_long.py` - Long videos with lower VRAM
  - `base.py` - Shared pipeline logic (latent merging, prompt extension, CPU offload)
- `models/` - Model definitions:
  - `wan_video_dit.py` - DiT architecture
  - `wan_video_vae.py` - VAE architectures (WanVideoVAE, Wan22VideoVAE, LightX2VVAE)
  - `model_manager.py` - Model loading with HuggingFace/civitai/diffusers support
- `vram_management/layers.py` - Tiling implementations for VAE and DiT

### Key Concepts
- **5 VAE Options**: Wan2.1, Wan2.2, LightVAE_W2.1, TAE_W2.2, LightTAE_HY1.5 (each maps to distinct file)
- **Attention Modes**: `sparse_sage` (recommended default), `flash_attention_2`, `sdpa`, `block_sparse`
- **VRAM Optimization**: Tiling (`tiled_vae`, `tiled_dit`), chunking (`frame_chunk_size`), resize factor
- **Model Auto-Download**: VAE files auto-download from HuggingFace if missing

### Configuration
- `model_paths.yaml` - Override default model search path (`ComfyUI/models/FlashVSR/`)
- `model_paths.yaml` config is cached in memory with thread-safe access

### VRAM Tiers
| VRAM | Mode | Key Settings |
|------|------|--------------|
| 24GB+ | `full`/`tiny` | No tiling, bf16 |
| 16GB | `tiny` | tiled_vae=True, keep_models_on_cpu |
| 12GB | `tiny` | tiled_vae=True, tiled_dit=True, fp16, sparse_sage |
| 8GB | `tiny-long` | All tiling required, chunk_size=16-32, resize_factor=0.5 |# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## keep on track
1. plan and execution results save into logs with the date and time as filename 
2. commit and push the modification 