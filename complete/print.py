import open3d as o3d
import numpy as np

# 讀取 PLY 點雲文件
file_path = "C:/Users/ASUS/Desktop/POINT/red/ArUco/point_cloud_00001.ply"
point_cloud = o3d.io.read_point_cloud(file_path)

# 打印點雲範圍
bounds = np.asarray(point_cloud.get_axis_aligned_bounding_box().get_box_points())
print("Point Cloud Bounds:\n", bounds)

# 建立座標系，調整大小
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# 顯示點雲與座標系
o3d.visualization.draw_geometries(
    [point_cloud, coordinate_frame],
    window_name="Point Cloud Viewer with Coordinate Frame",
    width=800, height=600
)
