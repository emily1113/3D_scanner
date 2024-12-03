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
pcd0 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00001.ply")
pcd1 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00002.ply")
pcd2 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00003.ply")
pcd3 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00004.ply")
pcd4 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00005.ply")
pcd5 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00006.ply")
pcd6 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00007.ply")
pcd7 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ArUco_1125/00008.ply")

# 為每個點雲分別設定不同顏色
pcd0.paint_uniform_color([1, 0, 0])  # 紅色
pcd1.paint_uniform_color([0, 1, 0])  # 綠色
pcd2.paint_uniform_color([0, 0, 1])  # 藍色
pcd3.paint_uniform_color([1, 1, 0])  # 黃色

# 套用旋轉
rotate_point_cloud(pcd0, angle_z=45)
rotate_point_cloud(pcd1, angle_z=90)
rotate_point_cloud(pcd2, angle_z=135)
rotate_point_cloud(pcd3, angle_z=180)

# 套用平移
# translate_point_cloud(pcd1, x=0, y=3, z=-60)
# translate_point_cloud(pcd2, x=-60, y=3, z=-62)
# translate_point_cloud(pcd3, x=-60, y=3, z=-20)

# 合併點雲
combined_pcd = pcd0 + pcd1 + pcd2 + pcd3

# 儲存合併後的點雲
o3d.io.write_point_cloud("C:/Users/ASUS/Desktop/POINT/red/combined_point_cloud.ply", combined_pcd)

print("合併的點雲已保存為 combined_point_cloud.ply")
