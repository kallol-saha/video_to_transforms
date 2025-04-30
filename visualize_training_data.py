import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from argparse import ArgumentParser

# Example usage: python visualize_training_data.py --data_path /home/jacinto/robot-grasp/data/taxposed_for_demos_20241230_173916/train

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


# with open("configs/global_config.yaml", "r") as config:
#     args = yaml.load(config, Loader=yaml.FullLoader)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, help='folder where the taxposed data is stored')

    args = parser.parse_args()
    data_files = [f for f in os.listdir(args.data_path) if f.endswith('.npz')]
    for file in data_files:
        print("File = ", file)
        data = np.load(os.path.join(args.data_path, file))
        pcd = data["clouds"]
        pcd_seg = data["classes"]
        goal_mask = data["masks"]

        action_mask = np.where(pcd_seg == 0)[0]
        anchor_mask = np.where(pcd_seg == 1)[0]

        # if pcd_seg[np.argmax(pcd[:, 2])] == 4:
        #     print(i)

        plot_pcd(pcd, pcd_seg)
        # plot_pcd(pcd[action_mask], pcd_seg[action_mask])
        # plot_pcd(pcd[anchor_mask], pcd_seg[anchor_mask])