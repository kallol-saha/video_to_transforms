import heapq
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

DIFF_THRESH = 3.            # What is the point track difference below which object is considered stationary
STATIONARY_THRESH = 4       # What is the minimum change in timesteps between different stationary frames for an object to be considered transformed

# def plot_pcd(pts3d):

#     pcd = np.zeros_like(pts3d, dtype = np.float64)
#     pcd[:, :] = pts3d[:, :]
#     pts_vis = o3d.geometry.PointCloud()
#     pts_vis.points = o3d.utility.Vector3dVector(pcd)

#     return [pts_vis]


def plot_pcd(pts3d, pcd_seg = None):

    pcd = np.zeros_like(pts3d, dtype = np.float64)
    pcd[:, :] = pts3d[:, :]
    
    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd)
    
    if pcd_seg is not None:    

        seg_ids = np.unique(pcd_seg)
        n = len(seg_ids)
        cmap = plt.get_cmap("tab10")
        id_to_color = {uid: cmap(i / n)[:3] for i, uid in enumerate(seg_ids)}
        colors = np.array([id_to_color[seg_id] for seg_id in pcd_seg])
        # print("Seg IDs = ", seg_ids)
        # print("Colors = ", id_to_color)
        pts_vis.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pts_vis])

class DataCollector:

    def __init__(self, folder_path):

        # Prepare data folder:
        self.folder_path = folder_path + "/"
        os.makedirs(self.folder_path, exist_ok=True)
        os.makedirs(self.folder_path + "train/", exist_ok=True)
        os.makedirs(self.folder_path + "test/", exist_ok=True)

        self.train_demos = len(os.listdir(self.folder_path + "train/"))
        self.test_demos = len(os.listdir(self.folder_path + "test/"))

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
    
    def get_transformation_between_points(self, P, Q):
        """
        P => (n, 3)
        Q => (n, 3)
        """

        # Compute centroids
        P_mean = np.mean(P, axis=0)
        Q_mean = np.mean(Q, axis=0)
        
        # Center the points
        P_centered = P - P_mean
        Q_centered = Q - Q_mean
        
        # Compute cross-covariance matrix
        H = P_centered.T @ Q_centered
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # Compute translation vector
        t = Q_mean - R @ P_mean
        
        # Form transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return R, t, T
    
    def ransac_find_transformation(self, P, Q, sample_size = 100, threshold=0.1, max_iterations=100, inlier_ratio=0.5):
        best_R = None
        best_t = None
        best_inliers = []
        n_points = len(P)
        
        for _ in range(max_iterations):
            # Randomly sample a subset of correspondences
            indices = np.random.choice(n_points, sample_size, replace=False)
            P_sample = P[indices]
            Q_sample = Q[indices]
            
            # Estimate transformation using the subset
            R, t, _ = self.get_transformation_between_points(P_sample, Q_sample)
            
            # Transform all points and calculate residuals
            P_transformed = (R @ P.T).T + t
            residuals = np.linalg.norm(P_transformed - Q, axis=1)
            
            # Determine inliers
            inliers = np.where(residuals < threshold)[0]
            
            # Keep the model with the most inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_R = R
                best_t = t
        
        # Recompute the transformation using all inliers
        if len(best_inliers) > 3:
            P_inliers = P[best_inliers]
            Q_inliers = Q[best_inliers]
            best_R, best_t, _ = self.get_transformation_between_points(P_inliers, Q_inliers)

        # Form transformation matrix
        T = np.eye(4)
        T[:3, :3] = best_R
        T[:3, 3] = best_t
        
        return T
        
    def transform_pcd(self, pcd, T):

        # Convert to homogeneous coordinates
        ones = np.ones((pcd.shape[0], 1))
        pcd_h = np.hstack((pcd, ones))  # (n, 4)

        # Apply transformation
        pcd_transformed_h = (T @ pcd_h.T).T  # (n, 4)
        
        # Convert back to Cartesian coordinates
        pcd_transformed = pcd_transformed_h[:, :3]

        return pcd_transformed
    
    def get_transform_sequence(self, indices, objects, object_tracks, pcd_sequence):
        """
        object_tracks => list of (frames, points, 2) arrays
        """

        # In object tracks last axis, the 0th index is width (1280), the 1st index is height (720)

        transforms = np.zeros((len(objects), 4, 4))
        
        for i in tqdm(range(len(indices) - 1)):

            track = object_tracks[objects[i]]       # Get the point track of the object being moved

            pts2d_before = np.round(track[indices[i]]).astype('int')
            pts2d_after = np.round(track[indices[i+1]]).astype('int')

            pts3d_before = pcd_sequence[indices[i], pts2d_before[:, 1], pts2d_before[:, 0]]
            pts3d_after = pcd_sequence[indices[i+1], pts2d_after[:, 1], pts2d_after[:, 0]]
            # I've done a visualization check on the 3d points before and after, we are good to go for svd !!

            # plot_pcd(np.concatenate([pts3d_before, pts3d_after], axis = 0))

            T = self.ransac_find_transformation(pts3d_before, pts3d_after, threshold = 0.02)
            # I've done a visualization check for the transform. But, TODO: needs to be robust to outliers (see result in progress report)
            
            # transformed_pcd = self.transform_pcd(pts3d_before, T)
            # seg = np.zeros((2 * pts3d_before.shape[0],), dtype = np.int64)
            # seg[pts3d_before.shape[0]:] = 1
            # plot_pcd(np.concatenate([transformed_pcd, pts3d_after], axis = 0), seg)

            transforms[i] = T[:]

        return transforms
    
    def prepare_initial_pcd(self, masks, pcd, vis_threshold = 1.):
        """
        masks => (num_objects, 720, 1280)
        """

        num_objects = masks.shape[0]
        
        initial_pcd = pcd.reshape((-1, 3))

        weights = np.arange(num_objects) + 1
        mask_sum = np.tensordot(weights, masks, axes = ([0], [0])) - 1      # Objects get 0 to n segmentation, everything else gets a -1
        initial_pcd_seg = mask_sum.reshape((-1, ))

        vis_mask = initial_pcd[:, -1] < vis_threshold

        pcd = initial_pcd[vis_mask]
        pcd_seg = initial_pcd_seg[vis_mask]

        # Downsample environment points to equal object points:
        env_mask = (pcd_seg == -1)
        num_obj_points = (pcd_seg != -1).sum()
        num_env_points = env_mask.sum()

        sample_env_mask = np.array(([True] * num_obj_points) + ([False] * (num_env_points - num_obj_points)))
        np.random.shuffle(sample_env_mask)

        env_mask[env_mask] = env_mask[env_mask] & sample_env_mask

        downsample_mask = env_mask | (pcd_seg != -1)
        pcd = pcd[downsample_mask]
        pcd_seg = pcd_seg[downsample_mask]

        # plot_pcd(pcd, pcd_seg)

        return pcd, pcd_seg
    
    def save_final_data(self, initial_pcd, initial_pcd_seg, transforms, objects, initial_rgb, mode = "train"):
        
        prev_pcd = initial_pcd.copy()
        pcd_seg = initial_pcd_seg.copy()
        
        for i in range(len(transforms)):

            obj_mask = (pcd_seg == objects[i])
            
            transformed_pcd = prev_pcd.copy()
            transformed_pcd[obj_mask] = self.transform_pcd(transformed_pcd[obj_mask], transforms[i])

            classes = np.where(obj_mask, 0, 1)

            # plot_pcd(transformed_pcd, pcd_seg)
            # plot_pcd(transformed_pcd, classes)
        
            # TODO: Remember to transform to robot frame (do we need this? Because taxposeD applies random transforms anyways)
        
            if mode == "train":
                np.savez(
                        self.folder_path + str(mode) + "/" + str(self.train_demos) + "_teleport_obj_points.npz",
                        clouds=transformed_pcd,
                        masks=pcd_seg,
                        classes=classes,
                    )
                self.train_demos += 1
                
            if mode == "test":
                np.savez(
                        self.folder_path + str(mode) + "/" + str(self.test_demos) + "_teleport_obj_points.npz",
                        clouds=transformed_pcd,
                        masks=pcd_seg,
                        classes=classes,
                    )
                self.test_demos += 1

            prev_pcd = transformed_pcd.copy()
    
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