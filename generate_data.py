from gsam_wrapper import GSAM2

object_names = "hand. rubber duck."             # First mask should always be hand
video_path = "assets/videos/duck/rgb_vid.mp4"
frame = 0
device = "cuda:1"

gsam2 = GSAM2(device)
masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks(object_names, video_path, frame)
gsam2.visualize(video_path, masks, confidences, labels, input_boxes, frame)

print("Done")
