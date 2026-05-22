#!/bin/bash
#
# Preprocess all videos in data/lowres for FlashVSR
# Usage: bash preprocess_all_videos.sh [scale] [align_multiple]
#

set -e

SCALE=4 #${1:-4}
ALIGN=16 #${2:-128}
INPUT_DIR="data/lowres"
OUTPUT_DIR="data/lowres_preprocessed"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Batch Preprocessing for FlashVSR"
echo "========================================"
echo "Scale factor: ${SCALE}x"
echo "Alignment: ${ALIGN}"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

# Count total videos
TOTAL=$(ls -1 "$INPUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Found $TOTAL videos to process"
echo ""

# Process each video
COUNT=0
for input_file in "$INPUT_DIR"/*.mp4; do
    COUNT=$((COUNT + 1))
    filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/${filename%.mp4}.mp4"

    echo "----------------------------------------"
    echo "[$COUNT/$TOTAL] Processing: $filename"
    echo "Output: $output_file"

    if [ -f "$output_file" ]; then
        echo "SKIP: Output already exists"
        continue
    fi

    python3.10 preprocess_for_flashvsr.py \
        --input "$input_file" \
        --output "$output_file" \
        --scale "$SCALE" \
        --align_multiple "$ALIGN"
    echo "Done: $output_file"
done

echo ""
echo "========================================"
echo "Batch Preprocessing Complete!"
echo "========================================"
echo "Output files saved to: $OUTPUT_DIR"
echo ""

# List output files
echo "Preprocessed videos:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "No output files found"
