# 20260512_194421 Batch Inference Plan

## Objective
Create a bash script to perform batch inference on all videos in `data/lowres/**`.

## Research
- Input directory: `data/lowres/` contains several `.mp4` files.
- Command: `cli_main.py` is the entry point for inference.
- Defaults chosen: `--scale 2`, `--mode tiny`, `--tiled_vae`, `--tiled_dit` (to ensure it runs on most GPUs).

## Changes
- Created `inference_all.sh`: A bash script that finds all videos in `data/lowres` and processes them one by one.

## Execution Result
- Script created at `inference_all.sh`.
- Ready for execution by user or automation.
