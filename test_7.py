import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_normals(points, k=15):
    """
    根據點雲資料計算每個點的法向量
    :param points: (N, 3) numpy 陣列，代表 N 個點的 x, y, z 座標
    :param k: 用於計算法向量的鄰域點數量
    :return: (N, 3) numpy 陣列，每一列為相對應點的法向量
    """
    n_points = points.shape[0]
    normals = np.zeros_like(points)
    
    # 建立 k 近鄰查詢模型
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # 對每個點計算其鄰域的協方差矩陣與特徵分解
    for i in range(n_points):
        neighbor_pts = points[indices[i]]
        mean = neighbor_pts.mean(axis=0)
        cov = np.dot((neighbor_pts - mean).T, (neighbor_pts - mean)) / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 選取最小特徵值所對應的特徵向量作為法向量
        normals[i] = eigenvectors[:, 0]
        
    return normals

# 讀取點雲
pcd = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply")
points = np.asarray(pcd.points)

# 利用自定義函數計算法向量
normals = compute_normals(points, k=15)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 根據點雲尺寸設定法向量線段長度比例
points_np = np.asarray(pcd.points)
normals_np = np.asarray(pcd.normals)
scale = np.linalg.norm(points_np.max(axis=0) - points_np.min(axis=0)) * 0.02

# 建立 LineSet 來表示法向量，每個點對應一條線段
line_points = []
lines = []
for i, (p, n) in enumerate(zip(points_np, normals_np)):
    line_points.append(p)
    line_points.append(p + n * scale)
    lines.append([2*i, 2*i+1])
    
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(line_points),
    lines=o3d.utility.Vector2iVector(lines)
)
# 設定線條顏色為紅色
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

# 顯示點雲與法向量線條
o3d.visualization.draw_geometries([pcd, line_set])
