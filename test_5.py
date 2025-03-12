import open3d as o3d
import numpy as np

# 指定檔案路徑
source_file = "C:/Users/ASUS/Desktop/POINT/red/FPFH/5/point_cloud_with_normals_cut_0.ply"
target_file = "C:/Users/ASUS/Desktop/POINT/red/FPFH/5/point_cloud_with_normals_cut_3.ply"

# 讀入點雲
pcd_source = o3d.io.read_point_cloud(source_file)
pcd_target = o3d.io.read_point_cloud(target_file)


# 取得所有法向量並計算平均法向量
normals_source = np.asarray(pcd_source.normals)
normals_target = np.asarray(pcd_target.normals)
mean_normal_source = normals_source.mean(axis=0)
mean_normal_target = normals_target.mean(axis=0)
mean_normal_source /= np.linalg.norm(mean_normal_source)
mean_normal_target /= np.linalg.norm(mean_normal_target)

# 計算來源法向量到目標法向量所需的旋轉
v = np.cross(mean_normal_source, mean_normal_target)
s = np.linalg.norm(v)
c = np.dot(mean_normal_source, mean_normal_target)

# 當兩向量接近平行時，直接使用單位矩陣
if s < 1e-8:
    R = np.eye(3)
else:
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s**2))

# 將來源點雲依據旋轉矩陣進行旋轉對齊
pcd_source.rotate(R, center=(0, 0, 0))

# 顯示對齊後的結果
o3d.visualization.draw_geometries([pcd_source, pcd_target])
