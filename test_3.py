import open3d as o3d
import numpy as np

# 讀取點雲檔案
pcd = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/rotated_point_cloud.ply")

# 獲取點雲的所有點的座標
points = np.asarray(pcd.points)

# 計算點雲的中心點
center = points.mean(axis=0)

# 計算每個點到中心點的歐氏距離
distances = np.linalg.norm(points - center, axis=1)

# 找到距離最小的點的索引
min_distance_index = np.argmin(distances)

# 找到離中心最近的點的座標
nearest_point = points[min_distance_index]

print("中心點:", center)
print("離中心最近的點座標:", nearest_point)
print("距離:", distances[min_distance_index])
