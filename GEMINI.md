# GEMINI.md - FlashVSR Integrated

This document provides instructions and context for working with the **FlashVSR Integrated** project, a high-performance Video Super Resolution solution for ComfyUI and standalone CLI.

## Project Overview

**FlashVSR Integrated** (also known as `ComfyUI-FlashVSR_Stable`) is an optimized implementation of FlashVSR, a diffusion-based video super-resolution model. It features intelligent VRAM management, supporting GPUs from 8GB to 24GB+, and integrates several VAE options (Wan2.1, Wan2.2, LightVAE, TAE) for optimal performance and quality.

### Core Technologies
- **Python/PyTorch**: Core deep learning framework.
- **ComfyUI**: Primary integration as a custom node.
- **FlashVSR**: The underlying diffusion model for video upscaling.
- **Sparse Sage Attention**: Optimized attention mechanism for faster processing and lower memory usage.
- **Triton**: Required for some optimized kernels (e.g., Flash Attention).

## Architecture & Directory Structure

- `__init__.py`: ComfyUI entry point (registers nodes).
- `nodes.py`: Contains the ComfyUI node definitions and mapping to core logic.
- `cli_main.py`: Standalone CLI implementation, mirroring ComfyUI node parameters.
- `src/`: Core source code.
    - `configs/`: Model and pipeline configurations.
    - `models/`: Model architecture definitions (DiT, VAE, TCDecoder).
    - `pipelines/`: Specialized pipelines for different processing modes (`tiny`, `tiny-long`, `full`).
    - `schedulers/`: Flow matching and diffusion schedulers.
    - `vram_management/`: Tiling and memory optimization layers.
- `models/`: Default directory for model weights.
- `workflow/`: Sample ComfyUI workflows and preview images.
- `model_paths.yaml`: Configuration file for customizing model storage locations.

## Building and Running

### Installation
1.  **Dependencies**: Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Triton (Older GPUs)**: For Turing or older GPUs (GTX 16/RTX 20 series), use a specific version:
    ```bash
    pip install -U triton<3.3.0  # Linux
    pip install -U triton-windows<3.3.0  # Windows
    ```

### Running as ComfyUI Node
1.  Clone into `ComfyUI/custom_nodes/`.
2.  Restart ComfyUI. The nodes will appear under the `FlashVSR` category.
3.  **Models**: DiT models should be in `ComfyUI/models/FlashVSR/`. VAE models auto-download from HuggingFace on first use.

### Running via CLI
The CLI mirrors the ComfyUI parameters 1:1.
```bash
# Basic 2x upscale
python cli_main.py --input input.mp4 --output output.mp4 --scale 2

# Low VRAM mode (8GB GPUs)
python cli_main.py --input input.mp4 --output output.mp4 --scale 2 \
    --vae_model LightVAE_W2.1 --tiled_vae --tiled_dit \
    --frame_chunk_size 20 --resize_factor 0.5
```

## Development Conventions

- **VRAM Optimization**: Always consider memory efficiency. Use tiling (`tiled_vae`, `tiled_dit`) and chunking (`frame_chunk_size`) for high-resolution or long videos.
- **Modes**:
    - `tiny`: Standard fast mode.
    - `tiny-long`: Optimized for long videos with lower VRAM.
    - `full`: Maximum quality, highest VRAM requirements.
- **Attention Kernels**: `sparse_sage_attention` is the recommended default for a balance of speed and memory.
- **Auto-Download**: The project handles model downloads automatically if they are missing from the specified model path.
- **Configuration**: Use `model_paths.yaml` to override the default model search paths without modifying the code.

## Key Files & Purpose
- `install_block_sparse_attention.py`: Script to install specialized attention dependencies.
- `src/models/wan_video_vae.py`: Implementation of various VAE architectures.
- `src/pipelines/flashvsr_full.py`: The main inference pipeline.
- `src/vram_management/layers.py`: Implementation of tiling logic for DiT and VAE.

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
3. when saving file into logs folder, need to use date and time and pr as filename