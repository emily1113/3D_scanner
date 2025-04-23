import os
import re
import time
from datetime import datetime  # 用於顯示系統時間
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.neighbors import NearestNeighbors  # 用於自訂法向量計算

# ------------------ 自訂體素下採樣函式 ------------------
def voxel_downsample(point_cloud, voxel_size):
    """
    體素下採樣：利用指定的體素大小對點雲進行下採樣處理，
    將同一體素中的多個點替換成該體素所有點的重心。
    """
    pts = np.asarray(point_cloud.points)
    min_bounds = pts.min(axis=0)
    idx = np.floor((pts - min_bounds) / voxel_size).astype(int)
    voxels = {}
    for i, key in enumerate(map(tuple, idx)):
        voxels.setdefault(key, []).append(pts[i])
    down = [np.mean(v, axis=0) for v in voxels.values()]
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(np.array(down))
    return down_pcd

# ------------------ 自訂法向量計算函式 ------------------
def compute_normals(points, k=15, view_point=np.array([0,0,0], dtype=np.float64), tolerance=1e-8):
    """
    根據點雲資料計算每個點的法向量，並統一取正方向。
    :param points: (N,3) numpy 陣列
    :param k: 鄰域搜索點數
    :param view_point: 參考視點
    :return: (N,3) 法向量陣列
    """
    n = points.shape[0]
    normals = np.zeros((n,3), dtype=np.float64)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, idx = nbrs.kneighbors(points)
    for i in range(n):
        neigh = points[idx[i]]
        mu = neigh.mean(axis=0)
        cov = (neigh - mu).T @ (neigh - mu) / k
        eigvals, eigvecs = np.linalg.eigh(cov)
        nvec = eigvecs[:, 0]
        norm = np.linalg.norm(nvec)
        if norm > tolerance:
            nvec /= norm
        else:
            nvec = np.zeros(3, dtype=np.float64)
        if np.dot(points[i] - view_point, nvec) < 0:
            nvec = -nvec
        normals[i] = nvec
    return normals

# ------------------ 單對處理函式 ------------------
def process_pair(src_file, tgt_file, voxel_size, thresh, downsample):
    src = o3d.io.read_point_cloud(src_file)
    tgt = o3d.io.read_point_cloud(tgt_file)
    # 自訂法向量估算（原始點雲）
    src_pts = np.asarray(src.points, dtype=np.float64)
    src.normals = o3d.utility.Vector3dVector(compute_normals(src_pts))
    tgt_pts = np.asarray(tgt.points, dtype=np.float64)
    tgt.normals = o3d.utility.Vector3dVector(compute_normals(tgt_pts))

    # 強制下採樣條件
    if len(src.points) > 500000 or len(tgt.points) > 500000:
        print("偵測大點雲，強制下採樣！")
        downsample = True

    # 下採樣
    if downsample:
        before = len(src.points)
        src = voxel_downsample(src, voxel_size)
        print(f"[Downsample src] {src_file}: {before} -> {len(src.points)}")
        before = len(tgt.points)
        tgt = voxel_downsample(tgt, voxel_size)
        print(f"[Downsample tgt] {tgt_file}: {before} -> {len(tgt.points)}")
        # 下採樣後重新計算法向量
        src_pts = np.asarray(src.points, dtype=np.float64)
        src.normals = o3d.utility.Vector3dVector(compute_normals(src_pts))
        tgt_pts = np.asarray(tgt.points, dtype=np.float64)
        tgt.normals = o3d.utility.Vector3dVector(compute_normals(tgt_pts))

    # ICP 精配
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=thresh,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    src.transform(result.transformation)
    merged = src + tgt
    merged.paint_uniform_color([1, 0, 0])
    return merged

# ------------------ 檔名產生 ------------------
def get_new_filename(f1, f2):
    m1 = re.match(r"(\d{5})_(\d{5})\.ply", f1)
    m2 = re.match(r"(\d{5})_(\d{5})\.ply", f2)
    if m1 and m2:
        return f"{m1.group(1)}_{m2.group(2)}.ply"
    return f"merged_{f1}_{f2}.ply"

# ------------------ 主程式 ------------------
if __name__ == "__main__":
    # 參數設定
    voxel_size = 0.5
    thresh = voxel_size * 1.5
    base_folder = r"C:/Users/ASUS/Desktop/POINT/red/furiren/processed/"
    records = []

    # 初始階段：Stage 2
    stage = 2
    current = os.path.join(base_folder, str(stage))
    next_stage = stage + 1
    next_folder = os.path.join(base_folder, str(next_stage))
    os.makedirs(next_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(current) if f.endswith('.ply')])

    for i in range(len(files) - 1):
        s = os.path.join(current, files[i])
        t = os.path.join(current, files[i+1])
        ds_flag = True  # 永遠下採樣
        print(f"Stage {stage} [{i}]: {files[i]} vs {files[i+1]} (downsample={ds_flag})")
        print("  開始時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        start = time.time()
        merged = process_pair(s, t, voxel_size, thresh, ds_flag)
        end = time.time()
        print("  結束時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        out_name = get_new_filename(files[i], files[i+1])
        out_path = os.path.join(next_folder, out_name)
        o3d.io.write_point_cloud(out_path, merged)
        records.append({
            "Stage": stage,
            "Filename": out_name,
            "PointCount": len(merged.points),
            "MergeTime(s)": end - start,
            "StartTime": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            "EndTime": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S")
        })

    # 後續階段迭代
    while True:
        stage = next_stage
        current = next_folder
        next_stage += 1
        next_folder = os.path.join(base_folder, str(next_stage))
        os.makedirs(next_folder, exist_ok=True)
        files = sorted([f for f in os.listdir(current) if f.endswith('.ply')])
        if len(files) <= 1:
            print("完成所有階段合併。")
            break

        for i in range(len(files) - 1):
            s_file = os.path.join(current, files[i])
            t_file = os.path.join(current, files[i+1])
            ds_flag = True  # 移除每3次下採樣，改為永久下採樣
            print(f"Stage {stage} [{i}]: {files[i]} vs {files[i+1]} (downsample={ds_flag})")
            print("  開始時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            start = time.time()
            merged = process_pair(s_file, t_file, voxel_size, thresh, ds_flag)
            end = time.time()
            print("  結束時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            out_name = get_new_filename(files[i], files[i+1])
            out_p = os.path.join(next_folder, out_name)
            o3d.io.write_point_cloud(out_p, merged)
            records.append({
                "Stage": stage,
                "Filename": out_name,
                "PointCount": len(merged.points),
                "MergeTime(s)": end - start,
                "StartTime": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
                "EndTime": datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S")
            })
