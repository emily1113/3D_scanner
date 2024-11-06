import open3d as o3d
import numpy as np

# 讀取多角度的點雲數據（假設已經拍攝好）
pcd_data = o3d.data.DemoICPPointClouds()
pcd_0  = o3d.io.read_point_cloud(pcd_data.paths[0])
pcd_90 = o3d.io.read_point_cloud(pcd_data.paths[1])
# pcd_0 = o3d.io.read_point_cloud("path/to/pcd_0.ply")
# pcd_90 = o3d.io.read_point_cloud("path/to/pcd_90.ply")
pcd_180 = o3d.io.read_point_cloud("path/to/pcd_180.ply")
pcd_270 = o3d.io.read_point_cloud("path/to/pcd_270.ply")

# 設定旋轉矩陣
trans_90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
trans_180 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
trans_270 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# 初始對齊
pcd_90.transform(trans_90)
pcd_180.transform(trans_180)
pcd_270.transform(trans_270)

# 精確配準
threshold = 1.0
icp = o3d.pipelines.registration.registration_icp

# 配準 pcd_90 到 pcd_0
result_90 = icp(pcd_90, pcd_0, threshold, np.identity(4))
pcd_90.transform(result_90.transformation)

# 配準 pcd_180 到 pcd_0
result_180 = icp(pcd_180, pcd_0, threshold, np.identity(4))
pcd_180.transform(result_180.transformation)

# 配準 pcd_270 到 pcd_0
result_270 = icp(pcd_270, pcd_0, threshold, np.identity(4))
pcd_270.transform(result_270.transformation)

# 合併點雲
pcd_combined = pcd_0 + pcd_90 + pcd_180 + pcd_270

# 儲存完整的點雲圖
o3d.io.write_point_cloud("path/to/complete_model.ply", pcd_combined)

# 可視化
o3d.visualization.draw([pcd_combined])