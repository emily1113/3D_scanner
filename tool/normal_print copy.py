import os
import time
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def compute_normals(points, k=15, view_point=np.array([0,0,0], dtype=np.float64), tolerance=1e-8):
    n_points = points.shape[0]
    normals = np.zeros_like(points, dtype=np.float64)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    for i in range(n_points):
        neighbor_pts = points[indices[i]]
        mean = neighbor_pts.mean(axis=0)
        cov = (neighbor_pts - mean).T @ (neighbor_pts - mean) / k
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        normal = eigenvecs[:, 0]

        norm_val = np.linalg.norm(normal)
        if norm_val > tolerance:
            normal /= norm_val
        else:
            normal = np.zeros(3, dtype=np.float64)

        if np.dot(points[i] - view_point, normal) < 0:
            normal = -normal

        normals[i] = normal

    return normals

def process_single_pointcloud(input_path, output_path,
                              k=15, view_point=np.array([0,0,0]), tolerance=1e-8):
    # 檔案存在性檢查
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"找不到檔案: {input_path}")

    # 計時開始
    start = time.time()

    # 讀取點雲
    pcd = o3d.io.read_point_cloud(input_path)
    pts = np.asarray(pcd.points)

    # 計算法向量
    normals = compute_normals(pts, k=k, view_point=view_point, tolerance=tolerance)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 寫出結果
    o3d.io.write_point_cloud(output_path, pcd)

    # 計時結束
    end = time.time()
    elapsed = end - start
    print(f"處理檔案：{os.path.basename(input_path)}，用時：{elapsed:.2f} 秒")
    print(f"輸出結果：{output_path}")

if __name__ == "__main__":
    src_file = r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00000.ply"
    dst_file = r"C:\Users\user\Desktop\PointCloud\red\test\1.ply"
    process_single_pointcloud(src_file, dst_file, k=30, view_point=np.array([0,0,0]))
