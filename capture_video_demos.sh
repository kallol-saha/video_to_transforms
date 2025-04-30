#!/bin/bash

# Example usage: ./capture_video_demos.sh --num_videos 15 --duration 6
# Example usage: ./capture_video_demos.sh --output_directory ~/robot-grasp/data/demos/videos --num_videos 15 --duration 6

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_directory) OUTPUT_DIR="$2"; shift ;;
        --num_videos) NUM_VIDEOS="$2"; shift ;;
        --duration) DURATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR=./data/demos_$timestamp/
fi

# Set default number of videos if not provided
if [ -z "$NUM_VIDEOS" ]; then
    NUM_VIDEOS=15
fi

# Set default duration if not provided
if [ -z "$DURATION" ]; then
    DURATION=6
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

for i in $(seq 1 $NUM_VIDEOS); do
    echo "Press any key to start capturing video $i..."
    read -n 1 -s
    for j in $(seq 4 -1 1); do
        echo "Starting in $j seconds..."
        sleep 1
    done
    VIDEO_DIR="$OUTPUT_DIR/$i"
    mkdir -p "$VIDEO_DIR"
    OUTPUT_PATH="$VIDEO_DIR/video.mkv"
    python capture_video.py --duration "$DURATION" --output_path "$OUTPUT_PATH"
    echo "Video $i saved to $OUTPUT_PATH"
done

echo "All videos have been captured."