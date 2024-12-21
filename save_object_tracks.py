from gsam_wrapper import GSAM2
from cotracker_wrapper import Cotracker3
from data_collector import DataCollector, plot_pcd
import torch
import os
import numpy as np
from tqdm import tqdm
import gc

# --------------------- USER PARAMS --------------------------- #
object_names = "blue plate. pink plate. red bowl. white cup"
num_objects = 4
video_folder = "assets/videos/table_bussing_four_objects"
num_videos = 20
start_from = 10
frame = 0
device = "cuda:1"
vis_threshold = 1.
# ------------------------------------------------------------- #

for v in tqdm(range(start_from, num_videos)):

    video_path = video_folder + "/rgb_vid_" + str(v) + ".mp4"
    tracks_path = video_folder + "/tracks_" + str(v)
    pcd_vid_path = video_folder + "/pcd_vid_" + str(v) + ".npy"
    os.makedirs(tracks_path, exist_ok=True)

    # Load point cloud data:
    pcd_sequence = np.load(pcd_vid_path)

    # Instantiate modules
    gsam2 = GSAM2(device)
    cotracker3 = Cotracker3(device)

    # GSAM inference
    masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks(object_names, video_path, frame)
    filtered_masks, _ = gsam2.filter_masks(masks, labels, num_objects)
    # gsam2.visualize(video_path, masks, confidences, labels, input_boxes, frame)

    del gsam2
    torch.cuda.empty_cache()
    gc.collect()

    # Cotracker inference
    filtered_masks = filtered_masks[:, 0]
    for i in range(filtered_masks.shape[0]):
        pred_tracks = cotracker3.get_tracks(video_path, filtered_masks[i])
        pred_tracks = pred_tracks.cpu().detach().numpy()
        np.save(tracks_path + "/" + str(i) + ".npy", pred_tracks)

    del cotracker3
    torch.cuda.empty_cache()

