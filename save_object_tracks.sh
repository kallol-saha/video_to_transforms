#!/bin/bash

TRACKS_SCRIPT="save_object_tracks.py"
DATA_SCRIPT="generate_data.py"
start=0
end=19      # inclusive

for ((i=start; i<=end; i++)); do
    echo "Saving object tracks for $i th video"
    python $TRACKS_SCRIPT --video_index $i
done

for ((i=start; i<=end; i++)); do
    echo "Generating for $i th video"
    python $DATA_SCRIPT --video_index $i
done