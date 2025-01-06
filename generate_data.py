from gsam_wrapper import GSAM2
from cotracker_wrapper import Cotracker3
from data_collector import DataCollector, plot_pcd
import torch
import os
import numpy as np
from tqdm import tqdm
import argparse

def main(args):

    # --------------------- USER PARAMS --------------------------- #
    object_names = "blue plate. pink plate. red bowl. white cup"
    num_objects = 4
    video_folder = "assets/videos/table_bussing_four_objects"
    data_path = "assets/data/table_bussing_four_objects"
    frame = 0
    vis_threshold = 1.      # NOTE: All points further than this distance from the camera are removed
    device = "cuda:1"
    # ------------------------------------------------------------- #
    # object_names = "rubber duck. blue box. wooden bowl"
    # video_path = "assets/videos/example1/rgb_vid.mp4"
    # data_path = "assets/data/example1"

    video_path = video_folder + "/rgb_vid_" + str(args.video_index) + ".mp4"
    tracks_folder = video_folder + "/tracks_" + str(args.video_index) + "/"
    # TODO: pcd_vid_path => this is a pre-saved (T, N, 3) numpy array, that was taken from the kinect
    pcd_vid_path = video_folder + "/pcd_vid_" + str(args.video_index) + ".npy"

    # Load point cloud data:
    pcd_sequence = np.load(pcd_vid_path)

    # Instantiate modules
    gsam2 = GSAM2(device)
    cotracker3 = Cotracker3(device)
    data_collector = DataCollector(data_path)

    # GSAM inference
    masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks(object_names, video_path, frame)
    print(labels)
    filtered_masks, _ = gsam2.filter_masks(masks, labels, num_objects)
    gsam2.visualize(video_path, masks, confidences, labels, input_boxes, frame)

    # Get the initial point cloud:
    initial_pcd, initial_pcd_seg = data_collector.prepare_initial_pcd(filtered_masks, pcd_sequence[0], vis_threshold)
    # plot_pcd(initial_pcd, initial_pcd_seg)

    object_tracks = []
    for i in range(filtered_masks.shape[0]):
        object_tracks.append(np.load(tracks_folder + str(i) + ".npy"))
        # cotracker3.visualize(video_path, torch.tensor(object_tracks[-1], device=device).unsqueeze(0), filename = "video_" + str(args.video_index) + "_" + str(i))

    # Detect movement:  (Can ignore this and just take initial and final frame)
    indices, objects = data_collector.detect_movement(object_tracks)    # indices => video frames where object is stationary

    # Get transform sequence:
    transforms = data_collector.get_transform_sequence(indices, objects, object_tracks, pcd_sequence)

    # Save the data:
    data_collector.save_final_data(initial_pcd, initial_pcd_seg, transforms, objects, mode = "train")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_index", required=True, help="index of the video to process")

    args = parser.parse_args()

    main(args)

