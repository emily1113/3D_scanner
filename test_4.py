import open3d as o3d
import numpy as np

# 讀取 PLY 點雲
point_cloud = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1_40/point_cloud_00000.ply")

# 建立 KD-Tree
kdtree = o3d.geometry.KDTreeFlann(point_cloud)

# 計算每點的鄰域密度
points = np.asarray(point_cloud.points)
density = []
radius = 0.05  # 鄰域半徑
for i, point in enumerate(points):
    [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
    density.append(len(idx))

density = np.array(density)

# 篩選高密度點
threshold = np.percentile(density, 50)  # 篩選前 10% 高密度點
high_density_points = points[density > threshold]

# 建立新點雲
high_density_cloud = o3d.geometry.PointCloud()
high_density_cloud.points = o3d.utility.Vector3dVector(high_density_points)

# 儲存高密度點群
o3d.io.write_point_cloud("high_density_points.ply", high_density_cloud)
print("高密度點群已儲存！")
