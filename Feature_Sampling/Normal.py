import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_normals(points, k=15, view_point=np.array([0, 0, 0])):
    """
    根據點雲資料計算每個點的法向量，並統一取正方向。
    :param points: (N, 3) numpy 陣列，代表 N 個點的 x, y, z 座標。
    :param k: 用於計算法向量的鄰域點數量。
    :param view_point: 參考視點，所有法向量都將統一指向該點的外側。
    :return: (N, 3) numpy 陣列，每一列為相對應點的法向量。
    """
    n_points = points.shape[0]
    normals = np.zeros_like(points)
    
    # 建立 k 近鄰查詢模型
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        neighbor_pts = points[indices[i]]
        mean = neighbor_pts.mean(axis=0)
        cov = np.dot((neighbor_pts - mean).T, (neighbor_pts - mean)) / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 選取最小特徵值對應的特徵向量作為法向量
        normal = eigenvectors[:, 0]
        # 統一法向量方向：使其與從視點到該點的向量同向
        if np.dot(points[i] - view_point, normal) < 0:
            normal = -normal
        normals[i] = normal
        
    return normals

# 讀取點雲（請根據實際路徑替換檔案名稱）
input_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply"
pcd = o3d.io.read_point_cloud(input_path)
points = np.asarray(pcd.points)

# 計算法向量，並設定參考視點（此處以原點為例，可依需求修改）
view_point = np.array([0, 0, 0])
normals = compute_normals(points, k=15, view_point=view_point)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 存檔方式一：存成包含法向量資訊的 PLY 檔案
output_ply_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_with_normals.ply"
o3d.io.write_point_cloud(output_ply_path, pcd)
print("點雲與法向量已存入：", output_ply_path)

# # 存檔方式二：存成 CSV 檔案，格式為 [x, y, z, nx, ny, nz]
# output_csv_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_normals.csv"
# data = np.hstack((points, normals))
# np.savetxt(output_csv_path, data, delimiter=",", header="x,y,z,nx,ny,nz", comments="")
# print("點雲與法向量資料已存入：", output_csv_path)

# ------------------------------
# 保留原本的顯示結果：利用 LineSet 顯示法向量，並將點雲顏色設為黑色
pcd.paint_uniform_color([0, 0, 0])  # 將所有點設為黑色

points_np = np.asarray(pcd.points)
normals_np = np.asarray(pcd.normals)
# 根據點雲尺寸決定法向量線段長度
scale = np.linalg.norm(points_np.max(axis=0) - points_np.min(axis=0)) * 0.02

line_points = []
lines = []
for i, (p, n) in enumerate(zip(points_np, normals_np)):
    line_points.append(p)
    line_points.append(p + n * scale)
    lines.append([2 * i, 2 * i + 1])
    
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(line_points),
    lines=o3d.utility.Vector2iVector(lines)
)
# 設定法向量線條顏色為紅色
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

# 顯示點雲與法向量線條（原本的顯示結果保留）
o3d.visualization.draw_geometries([pcd, line_set])
