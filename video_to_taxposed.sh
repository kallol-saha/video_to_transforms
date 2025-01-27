#!/bin/bash

# Example usage: ./video_to_taxposed.sh --input_dir ~/robot-grasp/data/demos/demos_20241230_173916 --output_dir ~/robot-grasp/data/taxposed_for_demos_20241230_173916 --object_names "mug." --debug

# Default values
input_dir=""
output_dir=""
object_names=""
debug=false  # Add default value for debug

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            input_dir="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --object_names)
            object_names="$2"
            shift 2
            ;;
        --debug)
            debug=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if required directories are provided
if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
    echo "Error: Both input_dir and output_dir must be specified"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Created output directory: $output_dir"
fi

# Activate miniconda
echo "Activating virtual environment..."
source ~/miniconda3/etc/profile.d/conda.sh
# Activate the conda environment
conda activate vid2trans

# Process each directory in input_dir
for dir in $(find "$input_dir" -maxdepth 1 -mindepth 1 -type d | sort); do
    echo "Processing directory: $dir"
    if [ "$debug" = true ]; then
        python video_to_taxposed.py --input_path "$dir" --output_path "$output_dir" --object_names "$object_names" --vis_threshold 2 --icp --debug
    else
        python video_to_taxposed.py --input_path "$dir" --output_path "$output_dir" --object_names "$object_names" --vis_threshold 2 --icp
    fi
done