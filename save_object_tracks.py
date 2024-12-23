from gsam_wrapper import GSAM2
from cotracker_wrapper import Cotracker3
from data_collector import DataCollector, plot_pcd
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import gc

def main(args):

    # --------------------- USER PARAMS --------------------------- #
    object_names = "blue plate. pink plate. red bowl. white cup"
    num_objects = 4
    video_folder = "assets/videos/table_bussing_four_objects"
    frame = 0
    device = "cuda:1"
    # ------------------------------------------------------------- #
    
    video_path = video_folder + "/rgb_vid_" + str(args.video_index) + ".mp4"
    tracks_path = video_folder + "/tracks_" + str(args.video_index)
    os.makedirs(tracks_path, exist_ok=True)

    # Instantiate modules
    gsam2 = GSAM2(device)
    cotracker3 = Cotracker3(device)

    # GSAM inference
    masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks(object_names, video_path, frame)
    filtered_masks, _ = gsam2.filter_masks(masks, labels, num_objects)
    # gsam2.visualize(video_path, masks, confidences, labels, input_boxes, frame)
    # TODO: Save the initial pcd and initial pcd seg here itself so that it doesn't GSAM doesn't need to be rerun again for data generation

    # Cotracker inference
    filtered_masks = filtered_masks[:, 0]
    for i in range(filtered_masks.shape[0]):
        pred_tracks = cotracker3.get_tracks(video_path, filtered_masks[i])
        pred_tracks = pred_tracks.cpu().detach().numpy()
        np.save(tracks_path + "/" + str(i) + ".npy", pred_tracks)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_index", required=True, help="index of the video to process")

    args = parser.parse_args()

    main(args)

