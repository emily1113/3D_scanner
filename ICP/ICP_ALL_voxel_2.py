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
    base_folder = r"C:\Users\user\Desktop\PointCloud\red\furiren_ALL"

    # 第一階段：對所有檔案進行下採樣與配準
    output_folder = os.path.join(base_folder, "2")
    os.makedirs(output_folder, exist_ok=True)

    total_files = 77
    for i in range(total_files - 1):
        source_file = os.path.join(base_folder, f"normals_point_cloud_{i:05d}.ply")
        target_file = os.path.join(base_folder, f"normals_point_cloud_{i+1:05d}.ply")
        if not os.path.exists(source_file) or not os.path.exists(target_file):
            print(f"檔案不存在，略過: {source_file} 或 {target_file}")
            continue

        print(f"第一階段配準 (always downsample): {source_file} 與 {target_file}")
        start_time = time.time()
        merged_pcd = process_pair(source_file, target_file, voxel_size, refined_distance_threshold, downsample=True)
        end_time = time.time()
        merge_time = end_time - start_time
        output_filename = f"{i:05d}_{i+1:05d}.ply"
        output_path = os.path.join(output_folder, output_filename)
        o3d.io.write_point_cloud(output_path, merged_pcd)
        print(f"儲存至: {output_path}，耗時: {merge_time:.2f} 秒")

    stage = 2

    # 後續階段：僅在階段為5的倍數時下採樣
    while True:
        current_folder = os.path.join(base_folder, str(stage))
        files = sorted([f for f in os.listdir(current_folder) if f.endswith(".ply")])
        num_files = len(files)
        print(f"階段 {stage} 檔案數： {num_files}")
        if num_files <= 1:
            print("只剩下一個檔案，結束合併！")
            break

        next_stage = stage + 1
        next_folder = os.path.join(base_folder, str(next_stage))
        os.makedirs(next_folder, exist_ok=True)

        # 判斷是否下採樣：當前階段是5的倍數
        ds_flag = (stage % 5 == 0)

        for file1, file2 in zip(files[:-1], files[1:]):
            source_file = os.path.join(current_folder, file1)
            target_file = os.path.join(current_folder, file2)

            print(f"階段 {stage} 配準: {source_file} 與 {target_file} (downsample={ds_flag})")
            start_time = time.time()
            merged_pcd = process_pair(source_file, target_file, voxel_size, refined_distance_threshold, downsample=ds_flag)
            end_time = time.time()
            merge_time = end_time - start_time
            new_filename = get_new_filename(file1, file2)
            output_path = os.path.join(next_folder, new_filename)
            o3d.io.write_point_cloud(output_path, merged_pcd)
            print(f"→ 輸出: {output_path}，耗時: {merge_time:.2f} 秒")

        stage = next_stage

    # 顯示最終結果
    final_folder = os.path.join(base_folder, str(stage))
    final_files = [f for f in os.listdir(final_folder) if f.endswith(".ply")]
    if final_files:
        final_file = os.path.join(final_folder, final_files[0])
        print("最終成果檔案:", final_file)
        print("配準結束時間：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # final_pcd = o3d.io.read_point_cloud(final_file)
        # o3d.visualization.draw_geometries([final_pcd], window_name="最終成果")
    else:
        print("無最終成果檔案可顯示。")

    print("所有階段合併完成！")
