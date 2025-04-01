import open3d as o3d
import numpy as np

# 指定點雲檔案的路徑
file_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/processed/20/00000_00019.ply"

# 讀取點雲資料
pcd = o3d.io.read_point_cloud(file_path)

# 印出點雲的基本資訊與點的數量
print("點雲資訊:", pcd)
print("點的數量:", np.asarray(pcd.points).shape[0])

# 使用 Open3D 的視覺化工具展示點雲
o3d.visualization.draw_geometries([pcd])
