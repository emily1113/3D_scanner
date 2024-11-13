import open3d as o3d
import numpy as np

# 讀取點雲檔案
pcd0 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/00.ply")
pcd1 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/01.ply")

# 為點雲分別設定顏色
pcd0.paint_uniform_color([1, 0, 0])  # 紅色
pcd1.paint_uniform_color([0, 1, 0])  # 綠色

# 建立 x 軸旋轉矩陣 (旋轉 135 度)
angle_x = np.deg2rad(135)  # x 軸角度轉換為弧度
rotation_matrix_x = np.array([
    [1, 0, 0],
    [0, np.cos(angle_x), -np.sin(angle_x)],
    [0, np.sin(angle_x), np.cos(angle_x)]
])

# 套用 x 軸旋轉矩陣
pcd1.rotate(rotation_matrix_x, center=(0, 0, 0))
pcd0.rotate(rotation_matrix_x, center=(0, 0, 0))

# 建立 y 軸旋轉矩陣 (逆時針旋轉 90 度)
angle_y = np.deg2rad(90)  # y 軸角度轉換為弧度
rotation_matrix_y = np.array([
    [np.cos(angle_y), 0, np.sin(angle_y)],
    [0, 1, 0],
    [-np.sin(angle_y), 0, np.cos(angle_y)]
])

# 套用 y 軸旋轉矩陣
pcd1.rotate(rotation_matrix_y, center=(0, 0, 0))

# # 定義平移向量，例如沿 z 軸平移 2 個單位
# translation_vector = np.array([350, 0, -500])

# # 套用平移到 pcd1
# pcd1.translate(translation_vector)

# 合併兩個點雲
combined_pcd = pcd0 + pcd1

# # 顯示合併後的點雲圖，顯示不同顏色
# o3d.visualization.draw_geometries([combined_pcd])

# 將合併後的點雲儲存為新檔案
o3d.io.write_point_cloud("combined_point_cloud.ply", combined_pcd)

print("合併的點雲已保存為 combined_point_cloud.ply")
