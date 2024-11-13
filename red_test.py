import open3d as o3d
import numpy as np

# 讀取點雲資料
pcd1 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00000.ply")
pcd2 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00001.ply")

# 下採樣點雲（選擇性步驟）
pcd1 = pcd1.voxel_down_sample(voxel_size=0.02)
pcd2 = pcd2.voxel_down_sample(voxel_size=0.02)

# 計算法線
pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 初步對齊（Z 軸逆時針旋轉 90 度）
theta = np.radians(90)  # 角度轉換為弧度
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

transformation_init = np.array([[cos_theta, -sin_theta, 0, 0],
                                [sin_theta,  cos_theta, 0, 0],
                                [0,          0,         1, 0],
                                [0,          0,         0, 1]])

pcd2.transform(transformation_init)

# 精細對齊 (ICP)
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd2, pcd1, max_correspondence_distance=0.05,
    init=transformation_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# 應用最佳轉換
pcd2.transform(reg_p2p.transformation)

# 可視化結果
o3d.visualization.draw_geometries([pcd1, pcd2])

# 顯示最佳轉換矩陣
print("Transformation Matrix:")
print(reg_p2p.transformation)

# 保存對齊後的點雲
o3d.io.write_point_cloud("aligned_point_cloud.ply", pcd2)
print("點雲已保存為 aligned_point_cloud.ply")
