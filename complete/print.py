import open3d as o3d

# 使用之前的點雲文件路徑
ply_path = "C:/Users/ASUS/Desktop/POINT/red/ArUco/point_cloud_00001.ply"

# 加載點雲
point_cloud = o3d.io.read_point_cloud(ply_path)

# 創建一個坐標系，設置大小以匹配點雲的比例
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

# 顯示點雲和坐標系
o3d.visualization.draw_geometries([point_cloud, coordinate_frame], window_name="Point Cloud with Origin and Axes")
