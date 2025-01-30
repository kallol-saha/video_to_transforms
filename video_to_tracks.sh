#!/bin/bash

# Example usage: ./video_to_tracks.sh input.mkv "cup" ./outputs 0 cuda
# Input video can be in .mkv and .mp4 format.

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <video_path> <object_name> [output_dir] [frame_number] [device]"
    echo "Example: $0 input.mkv \"blue cup\" ./outputs 0 cuda"
    exit 1
fi

# Parse arguments
VIDEO_PATH="$1"
OBJECT_NAME="$2"
OUTPUT_DIR="${3:-./outputs}"
FRAME_NUMBER="${4:-0}"
DEVICE="${5:-cuda}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define temporary mask file
MASK_FILE="$OUTPUT_DIR/temp_mask.npy"

# Step 1: Run GSAM to get mask
echo "Generating mask for object: $OBJECT_NAME"
python gsam_wrapper.py \
    --video_path "$VIDEO_PATH" \
    --object_name "$OBJECT_NAME." \
    --frame "$FRAME_NUMBER" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --debug

# Get the most recent mask file
LATEST_MASK=$(ls -t "$OUTPUT_DIR"/mask_*.png | head -n 1)

# Convert PNG mask to numpy array
python3 -c "
import cv2
import numpy as np
mask = cv2.imread('$LATEST_MASK', cv2.IMREAD_GRAYSCALE)
mask = (mask > 0).astype(np.float32)
np.save('$MASK_FILE', mask)
"

# Step 2: Run Cotracker with the mask
echo "Generating tracking video"
python cotracker_wrapper.py \
    --video_path "$VIDEO_PATH" \
    --mask_path "$MASK_FILE" \
    --output_path "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --output_name "tracked_${OBJECT_NAME// /_}"

# Cleanup temporary files
rm "$MASK_FILE"

echo "Processing complete. Output saved to: $OUTPUT_DIR"