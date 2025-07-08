import numpy as np
from tifffile import imread
import open3d as o3d

# 你的內參
fx = 2424.279
fy = 2424.682
cx = 963.017
cy = 621.057

# 讀取深度圖
depth = imread(r"C:\Users\ASUS\Desktop\POINT\red\furiren\depth_image_00000.tiff").astype(np.float32)  # shape: (H, W)
H, W = depth.shape

# 若單位是mm要換算成公尺，直接除以1000（根據你需求調整）
depth_m = depth

# 產生座標網格
u, v = np.meshgrid(np.arange(W), np.arange(H))
# 計算每個像素對應的三維點
X = (u - cx) * depth_m / fx
Y = (v - cy) * depth_m / fy
Z = depth_m

# 堆疊成點雲 (N, 3)
points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

# 移除 Z=0（沒有效的點）
valid = (Z > 0).reshape(-1)
points = points[valid]

# 生成 open3d 點雲
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("output.ply", pcd)
o3d.visualization.draw_geometries([pcd])
