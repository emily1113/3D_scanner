import os
import re
import time
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from datetime import datetime  # 🔹 新增：顯示系統時間用

# ------------------ 自訂體素下採樣函式 ------------------

def voxel_downsample(point_cloud, voxel_size):
    """
    體素下採樣：利用指定的體素大小對點雲進行下採樣處理，
    將同一體素中的多個點替換成該體素所有點的重心。
    """
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
    """
    使用自訂的 voxel_downsample 對 Open3D 點雲進行下採樣處理，
    並利用 compute_normals 計算下採樣後的法向量。
    
    參數:
      point_cloud: Open3D PointCloud 物件
      voxel_size: float，下採樣解析度
      
    返回:
      下採樣後，包含點與法向量資訊的 Open3D PointCloud 物件
    """
    points = np.asarray(point_cloud.points).astype(np.float64)
    down_points = voxel_downsample(points, voxel_size)
    normals = compute_normals(down_points, k=15, view_point=np.array([0, 0, 0], dtype=np.float64))
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(down_points)
    down_pcd.normals = o3d.utility.Vector3dVector(normals)
    return down_pcd

# ------------------ 配準與合併函式 ------------------

def refine_registration(source, target, initial_transformation, distance_threshold):
    """
    使用 ICP (點到平面) 進行精細配準
    """
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

def process_pair(source_file, target_file, voxel_size, refined_distance_threshold, downsample=True):
    """
    讀取兩個點雲檔案，
    根據 downsample 參數決定是否使用自訂體素下採樣（並計算法向量），
    再進行 ICP 配準，並合併配準後的點雲（最終統一著色為紅色）。
    
    若任一點雲點數超過 500,000，則強制下採樣。
    
    返回:
        合併後的 Open3D 點雲物件
    """
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
    """
    檔案命名規則：
    若 file1 為 "A_B.ply" 而 file2 為 "B_C.ply"，則回傳 "A_C.ply"
    """
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
    # 🔹 顯示匹配開始系統時間
    print("配準開始時間：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 參數設定
    voxel_size = 0.5  # 自訂體素下採樣解析度（依需求調整）
    refined_distance_threshold = voxel_size * 1.5
    base_folder = r"C:\Users\user\Desktop\PointCloud\red\furiren_ALL"
    records = []

    # ------------------ 第一階段 ------------------
    output_folder = os.path.join(base_folder, "2")
    os.makedirs(output_folder, exist_ok=True)
    total_files = 77
    for i in range(total_files - 1):
        source_file = os.path.join(base_folder, f"normals_point_cloud_{i:05d}.ply")
        target_file = os.path.join(base_folder, f"normals_point_cloud_{i+1:05d}.ply")
        if not os.path.exists(source_file) or not os.path.exists(target_file):
            print(f"檔案不存在，略過: {source_file} 或 {target_file}")
            continue
        print(f"第一階段配準: {source_file} 與 {target_file}")
        start_time = time.time()
        merged_pcd = process_pair(source_file, target_file, voxel_size, refined_distance_threshold, downsample=True)
        end_time = time.time()
        merge_time = end_time - start_time
        output_filename = f"{i:05d}_{i+1:05d}.ply"
        output_path = os.path.join(output_folder, output_filename)
        o3d.io.write_point_cloud(output_path, merged_pcd)
        print(f"儲存至: {output_path}，耗時: {merge_time:.2f} 秒")
        records.append({
            "Stage": 2,
            "Filename": output_filename,
            "PointCount": len(merged_pcd.points),
            "MergeTime(s)": merge_time
        })

    # ------------------ 後續階段迭代 ------------------
    stage = 2
    merge_counter = 1
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
        for i in range(num_files - 1):
            file1 = files[i]
            file2 = files[i+1]
            source_file = os.path.join(current_folder, file1)
            target_file = os.path.join(current_folder, file2)
            ds_flag = True if merge_counter == 1 or merge_counter % 3 == 0 else False
            print(f"階段 {stage} 配準: {source_file} 與 {target_file} (downsample={ds_flag})")
            start_time = time.time()
            merged_pcd = process_pair(source_file, target_file, voxel_size, refined_distance_threshold, downsample=ds_flag)
            end_time = time.time()
            merge_time = end_time - start_time
            new_filename = get_new_filename(file1, file2)
            output_path = os.path.join(next_folder, new_filename)
            o3d.io.write_point_cloud(output_path, merged_pcd)
            print(f"→ 輸出: {output_path}，耗時: {merge_time:.2f} 秒")
            records.append({
                "Stage": next_stage,
                "Filename": new_filename,
                "PointCount": len(merged_pcd.points),
                "MergeTime(s)": merge_time
            })
            merge_counter += 1
        stage = next_stage

    # ------------------ 顯示最終成果 ------------------
    final_folder = os.path.join(base_folder, str(stage))
    final_files = [f for f in os.listdir(final_folder) if f.endswith(".ply")]
    if final_files:
        final_file = os.path.join(final_folder, final_files[0])
        print("最終成果檔案:", final_file)
        final_pcd = o3d.io.read_point_cloud(final_file)
        o3d.visualization.draw_geometries([final_pcd], window_name="最終成果")
    else:
        print("無最終成果檔案可顯示。")

    print("所有階段合併完成！")
    # 🔹 顯示匹配結束系統時間
    print("配準結束時間：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ------------------ 輸出 Excel 記錄 ------------------
    df = pd.DataFrame(records, columns=["Stage", "Filename", "PointCount", "MergeTime(s)"])
    excel_log_file = os.path.join(base_folder, "merge_log.xlsx")
    df.to_excel(excel_log_file, index=False)
    print("Merge log saved to:", excel_log_file)
