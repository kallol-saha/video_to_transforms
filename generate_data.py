from gsam import GSAM2

object_names = "rubber duck."
video_path = "assets/videos/duck/rgb_vid.mp4"
frame = 0
device = "cuda:1"

gsam2 = GSAM2(device)

print("Done")
