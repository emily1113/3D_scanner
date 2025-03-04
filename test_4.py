import open3d as o3d
import numpy as np

# 讀取點雲資料
pcd = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply")

# 計算點雲的中心點
center = pcd.get_center()
print("點雲中心：", center)

# 估計法向量
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 將法向量調整為朝向點雲中心
# 注意：orient_normals_towards_camera_location 的參數表示「觀察點」位置，
#       這裡設為中心點可以讓法向量指向中心。
pcd.orient_normals_towards_camera_location(camera_location=center)

# 可視化結果，並顯示法向量
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
