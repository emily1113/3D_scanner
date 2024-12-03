import open3d as o3d
import numpy as np

# 加載點雲
ply_path = "C:/Users/ASUS/Desktop/POINT/red/ArUco/point_cloud_00001.ply"
point_cloud = o3d.io.read_point_cloud(ply_path)

# 將點雲轉換為 numpy 數組
points = np.asarray(point_cloud.points)

# 定義齊次矩陣以將相機原點轉換到指定位置
transformation_matrix = np.array([
    [-0.13577399,  0.98951506, -0.04924803, -0.04176586],
    [ 0.82222742,  0.08480928, -0.56280499, -0.08418158],
    [-0.55272732, -0.11690736, -0.82512131,  1.01194042],
    [ 0.0,          0.0,          0.0,          1.0       ]
])

# 將點雲轉換為齊次坐標並應用變換矩陣
points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
transformed_points_homogeneous = points_homogeneous.dot(transformation_matrix.T)

# 提取變換後的三維點坐標
transformed_points = transformed_points_homogeneous[:, :3]

# 創建新的點雲對象以顯示變換前和變換後的點雲
transformed_point_cloud = o3d.geometry.PointCloud()
transformed_point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

# 顯示原始和變換後的點雲
point_cloud.paint_uniform_color([1, 0, 0])  # 原始點雲為紅色
transformed_point_cloud.paint_uniform_color([0, 1, 0])  # 變換後的點雲為綠色
o3d.visualization.draw_geometries([point_cloud, transformed_point_cloud],
                                  window_name="原始和變換後的點雲",
                                  width=800, height=600)

# 將修改後的點雲保存到新的 PLY 文件
output_ply_path = "C:/Users/ASUS/Desktop/POINT/red/ArUco/point_cloud_transformed.ply"
o3d.io.write_point_cloud(output_ply_path, transformed_point_cloud)

print(f"點雲已成功移動到新的原點位置並保存為: {output_ply_path}")