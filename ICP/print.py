import open3d as o3d
import numpy as np

def get_camera_parameters_and_visualize(point_cloud_path):
    # 讀取點雲
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    # 創建可視化視窗
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    # 啟用視窗並允許用戶調整視角
    vis.run()

    # 獲取視角的相機參數
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    front = param.extrinsic[:3, 2].tolist()
    lookat = param.extrinsic[:3, 3].tolist()
    up = param.extrinsic[:3, 1].tolist()

    # 銷毀視窗
    vis.destroy_window()

    # 打印相機參數
    print("Front:", front)
    print("LookAt:", lookat)
    print("Up:", up)

    # 將相機位置標示在點雲中
    camera_position = np.array(lookat)  # 使用 lookat 位置作為相機位置
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    camera_sphere.translate(camera_position)
    camera_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 將相機位置設為紅色

    # 顯示點雲和相機位置
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(camera_sphere)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 替換為你的點雲檔案路徑
    point_cloud_path = "C:/Users/ASUS/Desktop/POINT/red/1118/point_cloud_00000.ply"
    get_camera_parameters_and_visualize(point_cloud_path)
