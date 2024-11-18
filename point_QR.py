import open3d as o3d
import numpy as np

# 讀取兩張點雲圖
pcd1_path = "C:/Users/ASUS/Desktop/POINT/red/1_40/point_cloud_00000.ply"
pcd2_path = "C:/Users/ASUS/Desktop/POINT/red/1_40/point_cloud_00001.ply"

pcd1 = o3d.io.read_point_cloud(pcd1_path)
pcd2 = o3d.io.read_point_cloud(pcd2_path)

# 提供的旋轉矩陣和平移向量
R = np.array([
    [-1.83624730e-01, -1.05426557e+00,  0.00000000e+00,  1.39734229e+03],
    [ 7.37693491e-01,  7.60346651e-02,  0.00000000e+00,  1.22089686e+01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]
])
T = np.array([[1], [1], [1], [1]])

# 截取有效的旋轉矩陣和平移向量
rotation_matrix = R[:3, :3]
translation_vector = R[:3, 3]

# 確保平移向量為 1D
translation_vector = translation_vector.reshape((3,))

# 對第二個點雲應用旋轉和平移
pcd2_points = np.asarray(pcd2.points)
transformed_points = np.dot(pcd2_points, rotation_matrix.T) + translation_vector

# 更新點雲數據
pcd2.points = o3d.utility.Vector3dVector(transformed_points)

# 保存旋轉和平移後的點雲
output_path = "transformed_point_cloud.ply"  # 替換為您想要保存的路徑
o3d.io.write_point_cloud(output_path, pcd2)

print(f"旋轉和平移後的點雲已保存到：{output_path}")
