import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_normals(points, k=15, view_point=np.array([0, 0, 0], dtype=np.float64), tolerance=1e-8):
    """
    根據點雲資料計算每個點的法向量，並統一取正方向，同時保證數值穩定性。
    
    :param points: (N, 3) numpy 陣列，資料類型為 np.float64，代表 N 個點的 x, y, z 座標。
    :param k: 用於計算法向量的鄰域點數量。
    :param view_point: 參考視點，所有法向量都將統一指向該點的外側，資料類型為 np.float64。
    :param tolerance: 當向量模長低於此容忍值時，認為向量為 0，避免除以 0 的情況。
    :return: (N, 3) numpy 陣列，每一列為正規化並統一方向後的法向量，資料類型為 np.float64。
    """
    n_points = points.shape[0]
    normals = np.zeros_like(points, dtype=np.float64)
    
    # 建立 k 近鄰查詢模型
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        neighbor_pts = points[indices[i]]
        mean = neighbor_pts.mean(axis=0)
        # 計算協方差矩陣
        cov = np.dot((neighbor_pts - mean).T, (neighbor_pts - mean)) / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 選取最小特徵值所對應的特徵向量作為法向量
        normal = eigenvectors[:, 0]
        
        # 正規化法向量，避免因模長過小導致數值不穩定
        norm_val = np.linalg.norm(normal)
        if norm_val > tolerance:
            normal = normal / norm_val
        else:
            normal = np.array([0, 0, 0], dtype=np.float64)
        
        # 統一法向量方向：使其與從視點到該點的向量同向
        if np.dot(points[i] - view_point, normal) < 0:
            normal = -normal
            
        normals[i] = normal
        
    return normals

# ---------------------------
# 讀取點雲並將資料轉為 double (np.float64)
input_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00006.ply"
pcd = o3d.io.read_point_cloud(input_path)
points = np.asarray(pcd.points).astype(np.float64)  # 轉為 double

# 設定參考視點（此處以原點為例，型態也轉為 np.float64）
view_point = np.array([0, 0, 0], dtype=np.float64)
normals = compute_normals(points, k=15, view_point=view_point)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 存檔方式一：存成包含法向量資訊的 PLY 檔案
output_ply_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/double_6.ply"
o3d.io.write_point_cloud(output_ply_path, pcd)
print("點雲與法向量已存入：", output_ply_path)

# # 存檔方式二：存成 CSV 檔案，格式為 [x, y, z, nx, ny, nz]
# output_csv_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_normals.csv"
# data = np.hstack((points, normals))
# np.savetxt(output_csv_path, data, delimiter=",", header="x,y,z,nx,ny,nz", comments="")
# print("點雲與法向量資料已存入：", output_csv_path)

# ---------------------------
# 保留原本的視覺化結果：將點雲設為黑色並用紅色線條顯示法向量

# 將點雲顏色設為黑色
pcd.paint_uniform_color([0, 0, 0])

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

# 顯示點雲與法向量線條
o3d.visualization.draw_geometries([pcd, line_set])
