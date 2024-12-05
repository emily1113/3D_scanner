import open3d as o3d
import numpy as np

# 定義旋轉函式
def rotate_point_cloud(pcd, angle_x=0, angle_y=0, angle_z=0, center=(0, 0, 0)):
    if angle_x != 0:
        angle_x_rad = np.deg2rad(angle_x)
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
            [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]
        ])
        pcd.rotate(rotation_matrix_x, center=center)

    if angle_y != 0:
        angle_y_rad = np.deg2rad(angle_y)
        rotation_matrix_y = np.array([
            [np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
            [0, 1, 0],
            [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]
        ])
        pcd.rotate(rotation_matrix_y, center=center)

    if angle_z != 0:
        angle_z_rad = np.deg2rad(angle_z)
        rotation_matrix_z = np.array([
            [np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
            [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
            [0, 0, 1]
        ])
        pcd.rotate(rotation_matrix_z, center=center)

# 定義平移函式
def translate_point_cloud(pcd, x=0, y=0, z=0):
    translation_vector = np.array([x, y, z])
    pcd.translate(translation_vector)

# 讀取點雲檔案
pcd0 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/point_cloud_00000 - Cloud.ply")
pcd1 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/point_cloud_00001 - Cloud.ply")
# pcd2 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1118_pcd/point_cloud_00002.pcd")
# pcd3 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1118_pcd/point_cloud_00003.pcd")

# 為每個點雲分別設定不同顏色
pcd0.paint_uniform_color([1, 0, 0])  # 紅色
pcd1.paint_uniform_color([0, 1, 0])  # 綠色
# pcd2.paint_uniform_color([0, 0, 1])  # 藍色
# pcd3.paint_uniform_color([1, 1, 0])  # 黃色

# 套用旋轉
rotate_point_cloud(pcd1, angle_z=90)
# rotate_point_cloud(pcd2, angle_z=180)
# rotate_point_cloud(pcd3, angle_z=270)

# 套用平移
translate_point_cloud(pcd1, x=-23, y=-95)
# translate_point_cloud(pcd2, x=72, y=-118)
# translate_point_cloud(pcd3, x=95, y=-23)

# 使用 ICP 將每個點雲依次與上一個對齊
def apply_icp(source, target):
    # 配準設定
    threshold = 1.0  # ICP配準距離閾值
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    # 應用變換矩陣對源點雲進行對齊
    source.transform(icp_result.transformation)
    return source

# 合併點雲的過程
combined_pcd = pcd0

# 將 pcd1 與 combined_pcd 進行配準
pcd1 = apply_icp(pcd1, combined_pcd)
combined_pcd += pcd1

# # 將 pcd2 與 combined_pcd 進行配準
# pcd2 = apply_icp(pcd2, combined_pcd)
# combined_pcd += pcd2

# # 將 pcd3 與 combined_pcd 進行配準
# pcd3 = apply_icp(pcd3, combined_pcd)
# combined_pcd += pcd3

# 儲存合併後的點雲
o3d.io.write_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1118/combined_point_cloud_icp.ply", combined_pcd)

print("合併的點雲已保存為 combined_point_cloud_icp.ply")
o3d.visualization.draw_geometries([combined_pcd], window_name="Combined Point Cloud with ICP")
