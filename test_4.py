import open3d as o3d

# 讀取點雲文件
pcd = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00001.ply")

# 可視化點雲
o3d.visualization.draw_geometries([pcd])

# 使用 KDTree 搜索參數估算法向量，這裡設定半徑和鄰域點數
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 顯示點雲及其法向量
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
