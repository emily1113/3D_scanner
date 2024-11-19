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
