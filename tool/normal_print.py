import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def compute_normals(points, k=15, view_point=np.array([0, 0, 0], dtype=np.float64), tolerance=1e-8):
    n_points = points.shape[0]
    normals = np.zeros_like(points, dtype=np.float64)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    for i in range(n_points):
        neighbor_pts = points[indices[i]]
        mean = neighbor_pts.mean(axis=0)
        cov = np.dot((neighbor_pts - mean).T, (neighbor_pts - mean)) / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]

        norm_val = np.linalg.norm(normal)
        if norm_val > tolerance:
            normal = normal / norm_val
        else:
            normal = np.array([0, 0, 0], dtype=np.float64)

        # 朝向view_point
        if np.dot(points[i] - view_point, normal) < 0:
            normal = -normal
        normals[i] = normal
    return normals

def main():
    # 輸入PLY路徑（自行修改檔案路徑）
    ply_path = r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\alpha_shell.ply"
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"讀取點雲，共 {points.shape[0]} 點")

    normals = compute_normals(points, k=15)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # 顯示點雲與法向量
    o3d.visualization.draw_geometries([pcd], 
        point_show_normal=True, 
        window_name="法向量視覺化"
    )

if __name__ == "__main__":
    main()
