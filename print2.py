import open3d as o3d

# 載入5個點雲
file_paths = [
    "C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00000.ply",
    "C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00001.ply",
    "C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00002.ply",
    "C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00003.ply",
    "C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00004.ply"
]

point_clouds = []
for path in file_paths:
    pcd = o3d.io.read_point_cloud(path)
    point_clouds.append(pcd)

# 顯示所有點雲
o3d.visualization.draw_geometries(point_clouds)
