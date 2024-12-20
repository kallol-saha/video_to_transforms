import heapq
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

DIFF_THRESH = 3.            # What is the point track difference below which object is considered stationary
STATIONARY_THRESH = 4       # What is the minimum change in timesteps between different stationary frames for an object to be considered transformed

def plot_pcd(pts3d):

    pcd = np.zeros_like(pts3d, dtype = np.float64)
    pcd[:, :] = pts3d[:, :]
    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd)

    return [pts_vis]

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
    
    def get_movement_order(self, object_change_points):

        min_heap = []
        for i, arr in enumerate(object_change_points):
            if arr:  # Only add if the array is not empty
                heapq.heappush(min_heap, (arr[0], i, 0))

        change_frames = []
        change_objects = []

        # Extract elements from the heap and keep track of which object was moved
        while min_heap:
            frame, object, is_in_array = heapq.heappop(min_heap)
            change_frames.append(frame)
            change_objects.append(object)

            # If there are more elements in the same array, push the next element into the heap
            if is_in_array + 1 < len(object_change_points[object]):
                next_value = object_change_points[object][is_in_array + 1]
                heapq.heappush(min_heap, (next_value, object, is_in_array + 1))

        return change_frames, change_objects
    
    def detect_movement(self, object_tracks):

        track_diffs = np.zeros((len(object_tracks), object_tracks[0].shape[0] - 1))
        object_change_points = []
        object_stationary_masks = []
        
        for i in range(len(object_tracks)):

            track = object_tracks[i]
            track_diff = np.linalg.norm(track[1:] - track[:-1], axis = -1)
            track_diff_avg = np.mean(track_diff, axis = -1)

            stationary_mask = np.where(track_diff_avg < DIFF_THRESH)[0]
            stationary_mask_diff = stationary_mask[1:] - stationary_mask[:-1]
            change_points = np.where(stationary_mask_diff > STATIONARY_THRESH)[0]

            object_change_points.append(change_points)
            object_stationary_masks.append(stationary_mask)

            track_diffs[i] = track_diff_avg

        change_frames, change_objects = self.get_movement_order(object_change_points)
        change_frames = np.array(change_frames)

        prev_change = 0
        indices = []

        for i in range(len(change_frames) + 1):

            if i == len(change_frames):
                indices.append((object_tracks[0].shape[0] + prev_change) // 2)      # After the last transition
                break
            
            change_start = object_stationary_masks[change_objects[i]][change_frames[i]]     # This is the start of the "peak" from the left, for the current object
            indices.append((change_start + prev_change) // 2)       # The midpoint of the valley is considered as the index

            prev_change = object_stationary_masks[change_objects[i]][change_frames[i]+1]    # Reset the previous change the end of the "peak" on the right 

        # Length of change_objects will always be one less than indices, because no more objects are moved at the last index.
        return np.array(indices), change_objects
    
    def get_transformation_between_points(self, pts1, pts2):

        pass
    
    def prepare_data(self, indices, objects, object_tracks, pcd_sequence):
        """
        object_tracks => list of (frames, points, 2) arrays
        """

        # In object tracks last axis, the 0th index is width (1280), the 1st index is height (720)
        
        for i in range(len(indices) - 1):

            track = object_tracks[objects[i]]       # Get the point track of the object being moved

            pts2d_before = np.round(track[indices[i]]).astype('int')
            pts2d_after = np.round(track[indices[i+1]]).astype('int')

            pts3d_before = pcd_sequence[indices[i], pts2d_before[:, 1], pts2d_before[:, 0]]
            pts3d_after = pcd_sequence[indices[i+1], pts2d_after[:, 1], pts2d_after[:, 0]]
            # I've done a visualization check on the 3d points before and after, we are good to go for svd !!

            # geometries = plot_pcd(np.concatenate([pts3d_before, pts3d_after], axis = 0))
            # o3d.visualization.draw_geometries(geometries)

            

            print("")
    
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