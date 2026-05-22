#!/bin/bash

# FlashVSR Batch Inference Script
# Processes all videos in data/lowres/

INPUT_DIR="data/lowres"
OUTPUT_DIR="outputs/upscaled_lowres"
SCALE=4
MODE="full"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting batch inference from $INPUT_DIR to $OUTPUT_DIR"
echo "Settings: Scale=$SCALE, Mode=$MODE"

# Iterate over all video files in the input directory
# Supports .mp4, .mkv, .avi, .mov
find "$INPUT_DIR" -maxdepth 2 -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" -o -name "*.mov" \) | while read -r vid; do
    filename=$(basename "$vid")
    output_path="$OUTPUT_DIR/${filename%.*}_upscaled_w8a16.mp4"

    if [ -f "$output_path" ]; then
        echo "------------------------------------------------"
        echo "Skipping: $filename (already exists at $output_path)"
        continue
    fi

    echo "------------------------------------------------"
    echo "Processing: $filename"
    
    python3.10 cli_main.py \
        --input "$vid" \
        --output "$output_path" \
        --scale "$SCALE" \
        --mode "$MODE" \
        --quantize_mode W8A16 \
        --ckpt_path models/FlashVSR-v1.1/diffusion_pytorch_model_w8a16.safetensors \
        --no_color_fix \
        --frame_chunk_size 30

    if [ $? -eq 0 ]; then
        echo "Successfully upscaled: $filename"
    else
        echo "Error processing: $filename"
    fi
done

echo "Batch inference complete."
