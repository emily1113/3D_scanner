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
        if len(source_pcd.points) > 600000 or len(target_pcd.points) > 600000:
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
    """
    只取 file1 的第一組數字，與 file2 的最後一組數字，
    合併成「00024_00027.ply」這種格式。
    """
    nums1 = re.findall(r'(\d+)', file1)
    nums2 = re.findall(r'(\d+)', file2)
    if nums1 and nums2:
        first = nums1[0]
        last = nums2[-1]
        return f"{first}_{last}.ply"
    # 若抓不到數字就退回合併名稱
    return f"merged_{file1}_{file2}.ply"



# ------------------ 主程序 ------------------

if __name__ == "__main__":
    print("One-pass merge start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    voxel_size = 0.5
    refined_distance_threshold = voxel_size * 1.5
    base_folder = r"C:\Users\user\Desktop\PointCloud\red\furiren_ALL"
    output_folder = os.path.join(base_folder, "stage_1")
    os.makedirs(output_folder, exist_ok=True)

    # 讀取所有檔案並排序
    files = sorted([f for f in os.listdir(base_folder) if f.endswith(".ply")])
    # 只做一次配對：0+1, 2+3, 4+5, …
    for i in range(0, len(files), 2):
        src_name = files[i]
        tgt_index = i + 1
        if tgt_index >= len(files):
            print(f"No pair for {src_name}, skip.")
            continue

        tgt_name = files[tgt_index]
        src_path = os.path.join(base_folder, src_name)
        tgt_path = os.path.join(base_folder, tgt_name)

        print(f"Merging {src_name} + {tgt_name} (downsample=True)")
        start = time.time()
        merged_pcd = process_pair(src_path, tgt_path,
                                  voxel_size, refined_distance_threshold,
                                  downsample=True)
        elapsed = time.time() - start

        # 產生只含第一檔與第二檔最後編號的純數字檔名
        out_name = get_new_filename(src_name, tgt_name)
        out_path = os.path.join(output_folder, out_name)
        o3d.io.write_point_cloud(out_path, merged_pcd)
        print(f" → Saved {out_name}, time: {elapsed:.2f}s")

    print("One-pass merge end:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

