import os

import numpy as np
import yaml

from data_collector import plot_pcd

# with open("configs/global_config.yaml", "r") as config:
#     args = yaml.load(config, Loader=yaml.FullLoader)

# demo = KeyboardDemo(args, -1)
path = "assets/data/table_bussing_four_objects/train/"

for i in range(39, len(os.listdir(path))):
    data = np.load(path + str(i) + "_teleport_obj_points.npz")
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