#!/bin/bash

# FlashVSR Batch Inference Script
# Processes all videos in data/lowres/

INPUT_DIR="data/lowres"
OUTPUT_DIR="outputs/upscaled_lowres"
SCALE=2
MODE="tiny"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting batch inference from $INPUT_DIR to $OUTPUT_DIR"
echo "Settings: Scale=$SCALE, Mode=$MODE"

# Iterate over all video files in the input directory
# Supports .mp4, .mkv, .avi, .mov
find "$INPUT_DIR" -maxdepth 2 -type f \( -name "*.mp4" -o -name "*.mkv" -o -name "*.avi" -o -name "*.mov" \) | while read -r vid; do
    filename=$(basename "$vid")
    output_path="$OUTPUT_DIR/${filename%.*}_upscaled.mp4"
    
    echo "------------------------------------------------"
    echo "Processing: $filename"
    
    python cli_main.py \
        --input "$vid" \
        --output "$output_path" \
        --scale "$SCALE" \
        --mode "$MODE" \
        --tiled_vae \
        --tiled_dit
        
    if [ $? -eq 0 ]; then
        echo "Successfully upscaled: $filename"
    else
        echo "Error processing: $filename"
    fi
done

echo "Batch inference complete."
