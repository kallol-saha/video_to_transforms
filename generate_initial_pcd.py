import argparse

from gsam_wrapper import GSAM2
from cotracker_wrapper import Cotracker3
from data_collector import DataCollector, plot_pcd_with_frame
import numpy as np
from tqdm import tqdm
from get_real_pcd import get_current_rgbd

# object_names = "rubber duck. blue box. wooden bowl"
img_path = "rgb.jpg"
pcd_path = "pcd.npy"
data_path = "assets/data/example1"
device = "cuda:1"
vis_threshold = 1.

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate initial point cloud")
    parser.add_argument("--object_names", required=True, type=str, help="Example: 'rubber duck. blue box. wooden bowl'")
    args = parser.parse_args()
    object_names = args.object_names
    objects = [item.strip() for item in object_names.split(".")]

    # Instantiate modules
    gsam2 = GSAM2(device)
    data_collector = DataCollector(data_path)

    pcd, _ = get_current_rgbd()

    # GSAM inference
    masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks_image(object_names, img_path)
    masks, labels = gsam2.filter_masks(masks, labels, len(objects))
    # Order the masks according to original object order
    indices = [labels.index(label) for label in objects]
    labels = [labels[i] for i in indices]
    masks = masks[indices]
    print(labels)
    print(indices)
    gsam2.visualize(img_path, masks, confidences, labels, input_boxes)

    # Get the initial point cloud:
    initial_pcd, initial_pcd_seg = data_collector.prepare_initial_pcd(masks, pcd, vis_threshold)

    np.save("initial_pcd.npy", initial_pcd)
    np.save("initial_pcd_seg.npy", initial_pcd_seg)

    plot_pcd_with_frame(initial_pcd, initial_pcd_seg)
