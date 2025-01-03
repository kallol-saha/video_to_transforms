# import open3d as o3d
# import numpy as np

# def create_arrow(origin, direction, color):
#     """Creates an arrow in Open3D."""
#     mesh = o3d.geometry.TriangleMesh.create_arrow(
#         cylinder_radius=0.02,
#         cone_radius=0.05,
#         cylinder_height=0.8,
#         cone_height=0.2,
#         resolution=20,
#     )
#     mesh.translate(origin)
#     mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(direction))
    
#     mesh.paint_uniform_color(color)
#     return mesh

# # Create the origin arrows
# origin_x = create_arrow([0, 0, 0], [np.pi/2, 0, 0], [1, 0, 0])
# origin_y = create_arrow([0, 0, 0], [0, np.pi/2, 0], [0, 1, 0])
# origin_z = create_arrow([0, 0, 0], [0, 0, 0], [0, 0, 1])

# # Create the translated and rotated arrows
# translation = [1, 2, 3]
# rotation = [np.pi/4, np.pi/6, np.pi/3]
# R = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
# print("R", R)

# translated_x = create_arrow(translation, [np.pi/2, 0, 0], [1, 0, 0])
# translated_x.rotate(R, center=[0, 0, 0])
# translated_y = create_arrow(translation, [0, np.pi/2, 0], [0, 1, 0])
# translated_y.rotate(R, center=[0, 0, 0])
# translated_z = create_arrow(translation, [0, 0, 0], [0, 0, 1])
# translated_z.rotate(R, center=[0, 0, 0])

# center_point = [0, 0, 0]

# # Create a sphere mesh with radius 0.5
# sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)

# # Translate the sphere to the desired center point
# sphere.translate(center_point)

# # Create a sphere mesh with radius 0.5
# sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)

# # Translate the sphere to the desired center point
# sphere2.translate(translation)

# # Visualize the arrows
# o3d.visualization.draw_geometries([
#     origin_x, origin_y, origin_z,
#     translated_x, translated_y, translated_z,
#     sphere, sphere2
# ])

import open3d as o3d
import numpy as np

# # Create two coordinate frames
# frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
# frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

# pcd_path = "/home/jacinto/video_to_transforms/initial_pcd.npy"

# CAMERA_CALIBRATION_FILE = "/home/jacinto/franka-panda-execution/extrinsics.npz"
# T_cam_to_world = np.load(CAMERA_CALIBRATION_FILE, allow_pickle=True)["T"]

# point_cloud_np = np.load(pcd_path)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

# # Define a transformation matrix
# # This example rotates the second frame by 45 degrees around the z-axis and translates it
# T = np.eye(4)

# best_grasp_camera_frame = np.array([[ 0.76935226,  0.63838094,  0.02380764, -0.06093371],
#  [-0.5294171,   0.6162904,   0.5830125,  -0.06333496],
#  [ 0.3575117,  -0.46114618,  0.81211424,  0.43073744],
#  [ 0,         0,         0,         1,      ]])

# best_grasp_world_frame = best_grasp_camera_frame @ T_cam_to_world.T

# # T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 4))
# T[:3, :3] = best_grasp_camera_frame[:3, :3]
# print(T)
# T[:3, 3] = best_grasp_camera_frame[:3, 3]

# # Apply transformation to the second frame
# frame2.transform(T)

# center_point = [0, 0, 0]
# sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
# sphere.translate(center_point)
# sphere.paint_uniform_color([1, 0, 1])


# center_point2 = [2, 0, 0]
# sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
# sphere2.translate(center_point2)
# sphere2.paint_uniform_color([1, 0, 1])

# # Visualize the frames
# o3d.visualization.draw_geometries([frame1, frame2, sphere, sphere2, pcd])


### world frame

import open3d as o3d
import numpy as np

# Create two coordinate frames
frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

pcd_path = "/home/jacinto/video_to_transforms/initial_pcd.npy"

CAMERA_CALIBRATION_FILE = "/home/jacinto/franka-panda-execution/extrinsics.npz"
T_cam_to_world = np.load(CAMERA_CALIBRATION_FILE, allow_pickle=True)["T"]


point_cloud_np = np.load(pcd_path)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
camera_extrinsics = np.load("/home/jacinto/franka-panda-execution/extrinsics2.npz")
pcd.transform(T_cam_to_world)

# Define a transformation matrix
# This example rotates the second frame by 45 degrees around the z-axis and translates it
T = np.eye(4)


best_grasp_camera_frame = np.array([[ 0.76935226,  0.63838094,  0.02380764, -0.06093371],
 [-0.5294171,   0.6162904,   0.5830125,  -0.06333496],
 [ 0.3575117,  -0.46114618,  0.81211424,  0.43073744],
 [ 0,         0,         0,         1,      ]])

# frame2.transform(best_grasp_camera_frame)

# T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 4))
T[:3, :3] = best_grasp_camera_frame[:3, :3]
print(T)
T[:3, 3] = best_grasp_camera_frame[:3, 3]

# Apply transformation to the second frame
# frame2.transform(T_cam_to_world)

All_T = T_cam_to_world @ best_grasp_camera_frame @ np.eye(4)
frame2.transform(All_T)
print("All_T", All_T)

center_point = [0, 0, 0]
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
sphere.translate(center_point)
sphere.paint_uniform_color([1, 0, 1])


center_point2 = [2, 0, 0]
sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
sphere2.translate(center_point2)
sphere2.paint_uniform_color([1, 0, 1])

# Visualize the frames
o3d.visualization.draw_geometries([frame1, frame2, sphere, sphere2, pcd])
