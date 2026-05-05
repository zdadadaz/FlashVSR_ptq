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
