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
    vid2="$OUTPUT_DIR/${filename%.*}_upscaledx4_w8a16.mp4"
    output_path="$OUTPUT_DIR/${filename%.*}_upscaledx4_w8a16_comb.mp4"

    if [ ! -f "$vid2" ]; then
        echo "------------------------------------------------"
        echo "Skipping: $filename (already exists at $output_path)"
        continue
    fi

    echo "------------------------------------------------"
    echo "Processing: $filename"
    
    # up-down
    # ffmpeg -i "$vid" -i "$vid2" -filter_complex \
    #         "[0:v]scale=iw*4:ih*4:flags=bicubic[top]; \
    #         [1:v][top]scale2ref=w=iw:h=ow/mdar:flags=bicubic[bottom][top_ref]; \
    #         [top_ref][bottom]vstack=inputs=2" \
    #         -c:v libx264 -crf 18 -preset veryfast "$output_path"

    # left-right
    ffmpeg -nostdin -y -i "$vid" -i "$vid2" -filter_complex \
    "[0:v]scale=iw*4:ih*4:flags=bicubic[left]; \
    [1:v][left]scale2ref=w=oh*mdar:h=ih:flags=bicubic[right][left_ref]; \
    [left_ref][right]hstack=inputs=2" \
    -r 30 -vsync 2 \
    -c:v libx264 -crf 18 -preset veryfast "$output_path"

    # break;
    if [ $? -eq 0 ]; then
        echo "Successfully upscaled: $filename"
    else
        echo "Error processing: $filename"
    fi
done

echo "Batch inference complete."
