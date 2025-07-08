import time
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

# ---------- 下採樣與法線計算 ----------
def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    min_b = points.min(axis=0)
    idxs = np.floor((points - min_b) / voxel_size).astype(int)
    vox = {}
    for pt, idx in zip(points, idxs):
        vox.setdefault(tuple(idx), []).append(pt)
    return np.array([np.mean(v, axis=0) for v in vox.values()])

def compute_normals(points: np.ndarray, k=15, vp=np.zeros(3), tol=1e-8) -> np.ndarray:
    nbr = NearestNeighbors(n_neighbors=k).fit(points)
    _, idxs = nbr.kneighbors(points)
    normals = np.zeros_like(points)
    for i, nei in enumerate(idxs):
        pts = points[nei]
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0]
        if np.linalg.norm(n) > tol:
            n /= np.linalg.norm(n)
        if np.dot(points[i] - vp, n) < 0:
            n = -n
        normals[i] = n
    return normals

def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points, dtype=np.float64)
    down_pts = voxel_downsample(pts, voxel_size)
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(down_pts)
    down_pcd.normals = o3d.utility.Vector3dVector(compute_normals(down_pts, k=15))
    return down_pcd

# ---------- 精細 ICP ----------
def refine_icp(src, tgt, max_dist):
    return o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

def merge_pair(pcd1, pcd2, voxel_size, downsample=True):
    # 保證法線存在
    for p in (pcd1, pcd2):
        if not p.has_normals():
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel_size*2, max_nn=30))
    # 下採樣
    if downsample:
        pcd1 = preprocess_pcd(pcd1, voxel_size)
        pcd2 = preprocess_pcd(pcd2, voxel_size)
    # 再次保證法線
    for p in (pcd1, pcd2):
        if not p.has_normals():
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel_size*2, max_nn=30))
    # 執行 ICP
    thresh = voxel_size * 1.5
    res = refine_icp(pcd1, pcd2, thresh)
    pcd1.transform(res.transformation)
    merged = pcd1 + pcd2
    merged.paint_uniform_color([1,0,0])
    return merged, res

if __name__ == "__main__":
    # 1. 指定要合併的四個點雲路徑
    files = [
        r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00000.ply",
        r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00001.ply",
        r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00002.ply",
        r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00003.ply"
    ]
    # 2. 參數設定
    voxel_size = 0.5
    downsample = True

    # 3. 依序讀取第一個點雲當作初始 merged_pcd
    merged_pcd = o3d.io.read_point_cloud(files[0])
    print(f"起始：{files[0]}")

    total_start = time.time()
    for idx in range(1, len(files)):
        next_pcd = o3d.io.read_point_cloud(files[idx])
        print(f">>> 合併 {idx}：上一階段 與 {files[idx]}")
        t0 = time.time()
        merged_pcd, res = merge_pair(merged_pcd, next_pcd, voxel_size, downsample)
        t1 = time.time()
        print(f"  時間：{t1-t0:.2f}s，RMSE={res.inlier_rmse:.4f}")

    total_end = time.time()
    # 4. 最終結果輸出
    output_path = r"C:\Users\user\Desktop\PointCloud\red\test\0_3.ply"
    o3d.io.write_point_cloud(output_path, merged_pcd)
    print(f"所有階段完成，共耗時 {total_end-total_start:.2f}s")
    print("結果存到：", output_path)
