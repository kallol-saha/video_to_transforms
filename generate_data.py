from gsam_wrapper import GSAM2
from cotracker_wrapper import Cotracker3
from data_collector import DataCollector
import numpy as np
from tqdm import tqdm

object_names = "rubber duck. blue box. wooden bowl"
video_path = "assets/videos/example1/rgb_vid.mp4"
data_path = "assets/data/example1"
frame = 0
device = "cuda:1"
vis_threshold = 1.

# Load point cloud data:
pcd_sequence = np.load("assets/videos/example1/pcd_vid.npy")
rgb_sequence = np.load("assets/videos/example1/rgb_vid.npy")

# Instantiate modules
gsam2 = GSAM2(device)
cotracker3 = Cotracker3(device)
data_collector = DataCollector(data_path)

# GSAM inference
masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks(object_names, video_path, frame)
gsam2.visualize(video_path, masks, confidences, labels, input_boxes, frame)

# Get the initial point cloud:
initial_pcd, initial_pcd_seg = data_collector.prepare_initial_pcd(masks, pcd_sequence)

# Cotracker inference
masks = masks[:, 0]
object_tracks = []
for i in tqdm(range(masks.shape[0])):
    pred_tracks, pred_visibility = cotracker3.get_tracks(video_path, masks[i])
    pred_tracks = pred_tracks.cpu().detach().numpy()
    pred_visibility = pred_visibility.cpu().detach().numpy()
    object_tracks.append(pred_tracks[0])
    # cotracker3.visualize(video_path, pred_tracks, pred_visibility, filename = str(i))

# Detect movement:
indices, objects = data_collector.detect_movement(object_tracks)

# Get transform sequence:
transforms = data_collector.get_transform_sequence(indices, objects, object_tracks, pcd_sequence)

# Save the data:
data_collector.save_final_data(initial_pcd, initial_pcd_seg, transforms, objects, rgb_sequence[0], mode = "train")

