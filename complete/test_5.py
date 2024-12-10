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
pcd0 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00000.ply")
pcd1 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply")
pcd2 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00002.ply")
pcd3 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00003.ply")

# 為每個點雲分別設定不同顏色
pcd0.paint_uniform_color([1, 0, 0])  # 紅色
pcd1.paint_uniform_color([0, 1, 0])  # 綠色
pcd2.paint_uniform_color([0, 0, 1])  # 藍色
pcd3.paint_uniform_color([1, 1, 0])  # 黃色

# # 套用旋轉
# # rotate_point_cloud(pcd0)
# rotate_point_cloud(pcd1, angle_y=5)
# rotate_point_cloud(pcd2, angle_y=10)
# rotate_point_cloud(pcd3, angle_y=15)

# 套用平移
# translate_point_cloud(pcd1, x=-60 )
# translate_point_cloud(pcd2, x=-120,z=5)
# translate_point_cloud(pcd3, x=-180,z=10)

# 合併點雲
combined_pcd = pcd0 + pcd2


# 儲存合併後的點雲
o3d.io.write_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1118/combined_point_cloud.ply", combined_pcd)

print("合併的點雲已保存為 combined_point_cloud.ply")
o3d.visualization.draw_geometries([combined_pcd], window_name="Combined Point Cloud")
