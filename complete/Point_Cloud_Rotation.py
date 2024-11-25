import numpy as np
import open3d as o3d

# 定義轉換矩陣
transformation_matrix = np.array([
    [-0.13577399,  0.98951506, -0.04924803, -0.04176586],
    [ 0.82222742,  0.08480928, -0.56280499, -0.08418158],
    [-0.55272732, -0.11690736, -0.82512131,  1.01194042],
    [ 0.0,          0.0,          0.0,          1.0       ]
])

# 讀取點雲
ply_path = "C:/Users/ASUS/Desktop/POINT/red/ArUco/point_cloud_00001.ply"
point_cloud = o3d.io.read_point_cloud(ply_path)
points = np.asarray(point_cloud.points)

# 將點雲轉換為齊次座標
ones = np.ones((points.shape[0], 1))
homogeneous_points = np.hstack((points, ones))

# 應用轉換矩陣
transformed_points_homogeneous = homogeneous_points @ transformation_matrix.T
transformed_points = transformed_points_homogeneous[:, :3]  # 提取前三列

# 更新點雲並保存
transformed_point_cloud = o3d.geometry.PointCloud()
transformed_point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

# 儲存轉換後的點雲
transformed_ply_path = "C:/Users/ASUS/Desktop/POINT/red/transformed_point_cloud.ply"
o3d.io.write_point_cloud(transformed_ply_path, transformed_point_cloud)

transformed_ply_path
