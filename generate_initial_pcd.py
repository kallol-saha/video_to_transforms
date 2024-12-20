from gsam_wrapper import GSAM2
from cotracker_wrapper import Cotracker3
from data_collector import DataCollector, plot_pcd
import numpy as np
from tqdm import tqdm

object_names = "rubber duck. blue box. wooden bowl"
img_path = "rgb.jpg"
pcd_path = "pcd.npy"
data_path = "assets/data/example1"
device = "cuda:1"
vis_threshold = 1.

# Instantiate modules
gsam2 = GSAM2(device)
data_collector = DataCollector(data_path)

pcd = np.load(pcd_path)

# GSAM inference
masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks_image(object_names, img_path)
gsam2.visualize(img_path, masks, confidences, labels, input_boxes)

# Get the initial point cloud:
initial_pcd, initial_pcd_seg = data_collector.prepare_initial_pcd(masks, pcd, vis_threshold)

np.save("initial_pcd.npy", initial_pcd)
np.save("initial_pcd_seg.npy", initial_pcd_seg)

plot_pcd(initial_pcd, initial_pcd_seg)