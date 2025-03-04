import open3d as o3d
import numpy as np

# 讀取點雲檔案
pcd = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply")
print("點數量：", np.asarray(pcd.points).shape[0])

# 計算法向量
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=30)

# 將點雲著色為黑色
pcd.paint_uniform_color([0, 0, 0])

# 建立法向量紅色線條的 LineSet
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
scale = 4  # 將線段縮放因子改為 0.2，讓法向量線段變長

line_points = []
lines = []
for i, (p, n) in enumerate(zip(points, normals)):
    start = p
    end = p + n * scale
    line_points.append(start)
    line_points.append(end)
    lines.append([2 * i, 2 * i + 1])

line_points = np.array(line_points)
lines = np.array(lines)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])

# 視覺化點雲與法向量線條
o3d.visualization.draw_geometries([pcd, line_set], point_show_normal=False)
