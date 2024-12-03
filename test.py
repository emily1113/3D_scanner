import open3d as o3d
import numpy as np

def get_camera_parameters_for_point_cloud(point_cloud):
    # 計算點雲的邊界框中心
    bounding_box = point_cloud.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()

    # 設置相機參數
    front = np.array([0.0, 0.0, 1.0])  # 相機前向向量，面向 Z 正方向  # 相機前向向量，面向 Z 負方向
    up = np.array([0.0, -1.0, 0.0])  # 相機上向向量，面向 Y 負方向
    lookat = center  # 相機觀察點設置為點雲中心

    # 回傳相機參數
    return front, lookat, up

# 使用例子
point_cloud = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00000.ply")
front, lookat, up = get_camera_parameters_for_point_cloud(point_cloud)
zoom = 0.5

o3d.visualization.draw_geometries([point_cloud],
                                  zoom=zoom,
                                  front=front.tolist(),
                                  lookat=lookat.tolist(),
                                  up=up.tolist())
