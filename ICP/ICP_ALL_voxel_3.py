import os
import re
import time
import numpy as np
import open3d as o3d
# import pandas as pd  # Excel 日誌功能已註解
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# ------------------ 自訂體素下採樣函式 ------------------

def voxel_downsample(point_cloud, voxel_size):
    min_bounds = np.min(point_cloud, axis=0)
    voxel_indices = np.floor((point_cloud - min_bounds) / voxel_size).astype(np.int32)

    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        voxel_dict.setdefault(key, []).append(point_cloud[i])

    downsampled_points = []
    for points in voxel_dict.values():
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        downsampled_points.append(centroid)

    return np.array(downsampled_points)


# ------------------ 自訂法向量計算函式 ------------------

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

        if np.dot(points[i] - view_point, normal) < 0:
            normal = -normal

        normals[i] = normal

    return normals


# ------------------ 點雲預處理函式 ------------------

def custom_preprocess_point_cloud(point_cloud, voxel_size):
    points = np.asarray(point_cloud.points).astype(np.float64)
    down_points = voxel_downsample(points, voxel_size)
    normals = compute_normals(down_points, k=15, view_point=np.array([0, 0, 0], dtype=np.float64))

    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(down_points)
    down_pcd.normals = o3d.utility.Vector3dVector(normals)
    return down_pcd


# ------------------ 配準與合併函式 ------------------

def refine_registration(source, target, initial_transformation, distance_threshold):
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp


def process_pair(source_file, target_file, voxel_size, refined_distance_threshold, downsample=True):
    source_pcd = o3d.io.read_point_cloud(source_file)
    target_pcd = o3d.io.read_point_cloud(target_file)

    if not source_pcd.has_normals():
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    if not downsample:
        if len(source_pcd.points) > 500000 or len(target_pcd.points) > 500000:
            print("發現點雲點數超過 500,000，強制下採樣！")
            downsample = True

    if downsample:
        src_before = len(source_pcd.points)
        source_proc = custom_preprocess_point_cloud(source_pcd, voxel_size)
        src_after = len(source_proc.points)
        print(f"[Custom Voxel Downsampling Source] {source_file}：{src_before} -> {src_after}")

        tgt_before = len(target_pcd.points)
        target_proc = custom_preprocess_point_cloud(target_pcd, voxel_size)
        tgt_after = len(target_proc.points)
        print(f"[Custom Voxel Downsampling Target] {target_file}：{tgt_before} -> {tgt_after}")
    else:
        source_proc = source_pcd
        target_proc = target_pcd

    if not source_proc.has_normals():
        source_proc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    if not target_proc.has_normals():
        target_proc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    initial_transformation = np.eye(4)
    result_icp = refine_registration(source_proc, target_proc, initial_transformation, refined_distance_threshold)
    source_proc.transform(result_icp.transformation)

    merged_pcd = source_proc + target_proc
    merged_pcd.paint_uniform_color([1, 0, 0])
    return merged_pcd


def get_new_filename(file1, file2):
    pattern = r"(\d{5})_(\d{5})\.ply"
    m1 = re.match(pattern, file1)
    m2 = re.match(pattern, file2)
    if m1 and m2:
        left = m1.group(1)
        right = m2.group(2)
        return f"{left}_{right}.ply"
    else:
        return f"merged_{file1}_{file2}.ply"


# ------------------ 主程序 ------------------

if __name__ == "__main__":
    print("配準開始時間：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    voxel_size = 0.5
    refined_distance_threshold = voxel_size * 1.5
    base_folder = r"C:\Users\user\Desktop\PointCloud\red\furiren_ALL\icp_2"
    stage = 1
    current_folder = base_folder

    while True:
        # 讀取當前階段所有 ply 檔
        files = sorted([f for f in os.listdir(current_folder) if f.endswith(".ply")])
        num_files = len(files)
        print(f"\n=== Stage {stage}: {num_files} 檔案 ===")
        if num_files <= 1:
            print("只剩下一個檔案，結束合併！")
            break

        # 決定下採樣開關：stage=1 必下採樣，否則 5 的倍數下採樣
        ds_flag = (stage == 1) or (stage % 10 == 0)

        # 準備下一階段資料夾
        next_stage = stage + 1
        next_folder = os.path.join(base_folder, str(next_stage))
        os.makedirs(next_folder, exist_ok=True)

        # 兩兩配對處理
        for file1, file2 in zip(files[:-1], files[1:]):
            src_path = os.path.join(current_folder, file1)
            tgt_path = os.path.join(current_folder, file2)
            print(f"Stage {stage} 配準: {file1} + {file2} (downsample={ds_flag})")

            t0 = time.time()
            merged = process_pair(src_path, tgt_path, voxel_size, refined_distance_threshold, downsample=ds_flag)
            t1 = time.time()

            # 依原檔名數字部分組新檔名
            out_name = get_new_filename(file1, file2)
            out_path = os.path.join(next_folder, out_name)
            o3d.io.write_point_cloud(out_path, merged)
            print(f"→ 輸出: {out_name}，耗時 {t1-t0:.2f} 秒")

        # 進入下一階段
        stage = next_stage
        current_folder = next_folder

    # 顯示最終結果
    final_files = [f for f in os.listdir(current_folder) if f.endswith(".ply")]
    if final_files:
        final_file = os.path.join(current_folder, final_files[0])
        print("最終成果檔案:", final_file)
    print("配準結束時間：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
