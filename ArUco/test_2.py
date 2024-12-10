import open3d as o3d
import numpy as np

# 載入點雲
point_cloud = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco/point_cloud_00001.ply")
points = np.asarray(point_cloud.points)

# 剛體變換矩陣
transformation_matrix = np.array([
    [-0.13577399,  0.98951506, -0.04924803, -0.12613289],
    [ 0.82222742,  0.08480928, -0.56280499, -0.25422837],
    [-0.55272732, -0.11690736, -0.82512131,  3.05605987],
    [ 0.          , 0.          ,  0.        ,  1.        ]
])

# 將點雲座標齊次化
points_h = np.hstack((points, np.ones((points.shape[0], 1))))

# 應用剛體變換矩陣
transformed_points_h = (transformation_matrix @ points_h.T).T
transformed_points = transformed_points_h[:, :3]

# 更新點雲座標
point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

# 保存轉換後的點雲
o3d.io.write_point_cloud("transformed_point_cloud.ply", point_cloud)
