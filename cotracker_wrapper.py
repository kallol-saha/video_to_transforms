import torch
import numpy as np
from cotracker3.cotracker.utils.visualizer import Visualizer
from tqdm import tqdm
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

        max_batch = 2000
        iters = queries.shape[1] // max_batch
        last_batch = queries.shape[1] % max_batch
        iters = iters + (1 if last_batch > 0 else 0)

        tracks_array = []
        visibility_array = []

        for i in tqdm(range(iters)):

            pred_tracks, pred_visibility = self.cotracker(video, queries = queries[:, i * max_batch : (i+1) * max_batch]) #grid_size=grid_size) # B T N 2,  B T N 1
            tracks_array.append(pred_tracks)
            visibility_array.append(pred_visibility)

        pred_tracks = torch.cat(tracks_array, dim=2)
        pred_visibility = torch.cat(visibility_array, dim=2)

        return pred_tracks, pred_visibility
    
    def visualize(self, video_path, pred_tracks, pred_visibility, filename = "video"):

        # pred_tracks => (B, frames, num_queries, 2) locations of the query points in each frame of the video
        # pred_visibility => (B, frames, num_queries) mask of whether the point is visible in that frame or not

        frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.device)  # B T C H W
        
        vis = Visualizer(save_dir="./outputs", pad_value=120, linewidth=3)
        vis.visualize(video, pred_tracks, pred_visibility, filename = filename) #, segm_mask = mask)