from gsam_wrapper import GSAM2
from cotracker_wrapper import Cotracker3

object_names = "rubber duck."
video_path = "assets/videos/duck/rgb_vid.mp4"
frame = 0
device = "cuda:1"

# Instantiate modules
gsam2 = GSAM2(device)
cotracker3 = Cotracker3(device)

# GSAM inference
masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks(object_names, video_path, frame)
gsam2.visualize(video_path, masks, confidences, labels, input_boxes, frame)

# Cotracker inferences
masks = masks[0, 0]     # TODO: Generalize to multi-object
pred_tracks, pred_visibility = cotracker3.get_tracks(video_path, masks)
cotracker3.visualize(video_path, pred_tracks, pred_visibility)

print("Done")
