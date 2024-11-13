# import open3d as o3d
# import numpy as np

# # 讀取點雲檔案
# pcd0 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00000.ply")
# pcd1 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00001.ply")
# # 建立旋轉矩陣 (x 軸旋轉 135 度)
# angle = np.deg2rad(135)  # 將角度轉換為弧度
# rotation_matrix = np.array([
#     [1, 0, 0],
#     [0, np.cos(angle), -np.sin(angle)],
#     [0, np.sin(angle), np.cos(angle)]
# ])

# # 套用旋轉矩陣
# pcd0.rotate(rotation_matrix, center=(0, 0, 0))
# pcd1.rotate(rotation_matrix, center=(0, 0, 0))

# # 可選：將旋轉後的點雲圖儲存為新檔案
# o3d.io.write_point_cloud("point_cloud_00000.ply", pcd0)
# o3d.io.write_point_cloud("point_cloud_00001.ply", pcd1)
# combined_pcd = pcd0 + pcd1
# o3d.io.write_point_cloud("0001.ply", combined_pcd)


import open3d as o3d
import numpy as np

# 讀取 .ply 檔案
ply_file_path = "C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00000.ply"  # 替換為你的檔案路徑
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# 確認座標
if point_cloud.is_empty():
    print("無法讀取座標，請檢查檔案內容或格式。")
else:
    points = np.asarray(point_cloud.points)
    print("座標資料：", points)
