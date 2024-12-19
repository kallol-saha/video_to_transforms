import numpy as np

DIFF_THRESH = 2.            # What is the point track difference below which object is considered stationary
STATIONARY_THRESH = 4       # What is the minimum change in timesteps between different stationary frames for an object to be considered transformed

class DataCollecter:

    def __init__(self):

        pass

    def get_indices(self, pred_tracks):
        """
        pred_tracks => (frames, num_points, 2)
        pred_visibility => (frame, num_points)
        """

        # TODO: Also use the visiblity mask if it helps

        track_diff = np.linalg.norm(pred_tracks[1:] - pred_tracks[:-1], axis = -1)
        track_diff_avg = np.mean(track_diff, axis = -1)

        stationary_mask = np.where(track_diff_avg < DIFF_THRESH)[0]
        stationary_mask_diff = stationary_mask[1:] - stationary_mask[:-1]

        change_points = np.where(stationary_mask_diff > STATIONARY_THRESH)[0]
        change_segs = np.empty(len(change_points)*2, dtype = stationary_mask.dtype)    # Elements are segment_start, segment_end for every ith and (i+1)th element
        change_segs[0::2] = stationary_mask[change_points]
        change_segs[1::2] = stationary_mask[change_points + 1]

        change_segs = np.concatenate(([0], change_segs, [len(track_diff_avg) - 1]))

        indices = []

        for i in range(0, len(change_segs), 2):

            stationary_segment = track_diff_avg[change_segs[i] : change_segs[i+1]]
            indices.append(np.argmin(stationary_segment) + change_segs[i])

        return indices

    def visualize(self):

        pcd_sequence = pcd_sequence[indices]
        rgb_sequence = rgb_sequence[indices]

        for i in range(pcd_sequence.shape[0]):

            pcd_vis = np.reshape(pcd_sequence[i], (-1, 3))
            vis_mask = pcd_vis[:, -1] < 1.

            pcd_vis = pcd_vis[vis_mask]
            rgb_vis = (np.reshape(rgb_sequence[i], (-1, 3)) / 255.)[:, [2, 1, 0]]
            rgb_vis = rgb_vis[vis_mask]
            
            pts_vis = o3d.geometry.PointCloud()
            pts_vis.points = o3d.utility.Vector3dVector(pcd_vis)
            pts_vis.colors = o3d.utility.Vector3dVector(rgb_vis)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0, 0, 0])

            o3d.visualization.draw_geometries([pts_vis, frame])

            # TODO: Index into the pcd_sequence with the pred_track and mask with the pred_visibility to get only the visible points across all stop indices.
            # TODO: Do an SVD between each of these points to get the optimal rotation and translation between them, and thus the transform
            # TODO: Store the transforms and the initial filtered point cloud as data.