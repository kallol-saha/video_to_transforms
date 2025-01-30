import torch
import numpy as np
from cotracker3.cotracker.utils.visualizer import Visualizer
from tqdm import tqdm
import argparse
import imageio.v3 as iio

class Cotracker3:

    def __init__(self, device):

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Run Offline CoTracker:
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)

    def get_tracks(self, video_path, mask):

        frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.device)  # B T C H W
        
        mask = torch.tensor(mask, device=self.device)#.unsqueeze(0).unsqueeze(0)
        queries = torch.nonzero(mask.t() == 1.0, as_tuple=False).to(torch.float32)

        # Concatenate the zeros column with the original tensor
        queries = torch.cat((torch.zeros((queries.size(0), 1), dtype=queries.dtype, device=queries.device), 
                            queries), 
                            dim=1)
        queries = queries.unsqueeze(0)

        # - queries. Queried points of shape (B, N, 3) in format (t, x, y) for frame index and pixel coordinates.

        max_batch = 400
        iters = queries.shape[1] // max_batch
        last_batch = queries.shape[1] % max_batch
        iters = iters + (1 if last_batch > 0 else 0)

        # tracks_array = []
        # visibility_array = []
        tracks = torch.zeros((video.shape[1], queries.shape[1], 2))

        for i in tqdm(range(iters)):

            pred_tracks, _ = self.cotracker(video, queries = queries[:, i * max_batch : (i+1) * max_batch], grid_size=3) # B T N 2,  B T N 1
            tracks[:, i * max_batch : (i+1) * max_batch] = pred_tracks[0]
            del pred_tracks
            torch.cuda.empty_cache()
            # pred_tracks => (frames, num_points, 2)
            
            # track_diff = torch.norm(pred_tracks[1:] - pred_tracks[:-1], dim=-1)
            # del pred_tracks
            # torch.cuda.empty_cache()
            # track_diff_sum = torch.sum(track_diff, dim=-1)
            # del track_diff
            # torch.cuda.empty_cache()

            # tracks_array.append(track_diff_sum)
            # visibility_array.append(pred_visibility)

        # output = torch.stack(tracks_array, dim=0)
        # output = torch.sum(output, dim = 0) / queries.shape[1]
        # pred_visibility = torch.cat(visibility_array, dim=2)

        return tracks
    
    def visualize(self, video_path, pred_tracks, output_path="./outputs", pred_visibility = None, filename = "video"):

        # pred_tracks => (B, frames, num_queries, 2) locations of the query points in each frame of the video
        # pred_visibility => (B, frames, num_queries) mask of whether the point is visible in that frame or not

        frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.device)  # B T C H W
        
        vis = Visualizer(save_dir=output_path, pad_value=120, linewidth=3)
        vis.visualize(video, pred_tracks, pred_visibility, filename = filename) #, segm_mask = mask)


if __name__ == "__main__":

    # Example usage: python cotracker_wrapper.py --video_path ./inputs/vid.mp4 --mask_path ./inputs/mask.npy --output_path ./outputs

    parser = argparse.ArgumentParser(description='Track points in a video using Cotracker3')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video. Can be mkv, mp4')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to binary mask file (numpy array or png)')
    parser.add_argument('--output_path', type=str, default='./outputs', help='Path to output directory')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cuda/cpu)')
    parser.add_argument('--output_name', type=str, default='video', help='Output filename')
    args = parser.parse_args()

    # Load mask from file
    if args.mask_path.endswith('.npy'):
        mask = np.load(args.mask_path)
    elif args.mask_path.endswith('.png'):
        mask = iio.imread(args.mask_path)
    else:
        raise ValueError("Unsupported mask file format. Please provide a .npy or .png file.")
    

    # Initialize tracker
    tracker = Cotracker3(device=args.device)

    # Get tracks
    tracks = tracker.get_tracks(args.video_path, mask)

    # Visualize results
    tracker.visualize(args.video_path, tracks.unsqueeze(0), output_path=args.output_path, filename=args.output_name)