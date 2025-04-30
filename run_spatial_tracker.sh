export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# python demo.py --model spatracker --downsample 1 --root ./assets --vid_name sintel_bandage \
#     --len_track 1 --fps_vis 15  --fps 1 --grid_size 60 --gpu 0 --rgbd

python SpaTracker/demo.py --model spatracker --downsample 1 --root /home/jacinto/robot-grasp/data/demos/simple_movements/4 --vid_name video \
    --len_track 1 --fps_vis 15  --fps 1 --grid_size 120 --gpu 0 --rgbd


# python demo.py --model spatracker --downsample 1 --vid_name task_dynamic_tossing \
    # --len_track 0 --fps_vis 15  --fps 2 --grid_size 20 --gpu 0 "$@"