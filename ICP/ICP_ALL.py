import os
import re
import time
import numpy as np
import open3d as o3d
import pandas as pd  # 用於寫入 Excel

# ------------------ 基本函式 ------------------

def load_point_cloud(file_path):
    """
    讀取點雲檔案，回傳 Open3D 點雲物件
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def preprocess_point_cloud(point_cloud, sample_step):
    """
    均勻下採樣（利用 numpy 陣列切片，每隔 sample_step 個點取一個）
    同時保留法向量與顏色資訊
    """
    points = np.asarray(point_cloud.points)
    downsampled_points = points[::sample_step]
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        downsampled_normals = normals[::sample_step]
        downsampled_pcd.normals = o3d.utility.Vector3dVector(downsampled_normals)
    
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        downsampled_colors = colors[::sample_step]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)
    
    return downsampled_pcd

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

def process_pair(source_file, target_file, sample_step, voxel_size, refined_distance_threshold, downsample=True):
    """
    讀取兩個點雲檔、估算法向量，根據 downsample 參數決定是否進行下採樣，
    再進行 ICP 配準，並合併配準後的點雲 (並統一著色為紅色)
    回傳合併後的點雲 (Open3D 格式)

    當 downsample 為 True 時，會顯示下採樣前後的點數量訊息。
    另外，若任一點雲的點數超過 500,000，則也會強制進行一次下採樣。

    參數:
      downsample: 若為 True 則對點雲進行下採樣；否則保留原始點雲，但當點數超過 500,000 時仍會下採樣
    """
    # 讀取原始點雲
    source_pcd = load_point_cloud(source_file)
    target_pcd = load_point_cloud(target_file)
    
    # 若點雲缺少法向量則估計
    if not source_pcd.has_normals():
        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

    # 檢查點數，若任一點雲超過 500,000 點，則強制下採樣
    if not downsample:
        if len(source_pcd.points) > 500000 or len(target_pcd.points) > 500000:
            print("發現點雲點數超過 500,000，強制下採樣！")
            downsample = True

    # 根據 downsample 標記決定是否進行下採樣
    if downsample:
        # 顯示下採樣前後點數量（來源點雲）
        src_before = len(source_pcd.points)
        source_proc = preprocess_point_cloud(source_pcd, sample_step)
        src_after = len(source_proc.points)
        print(f"[Downsampling Source] {source_file}：{src_before} -> {src_after}")
        
        # 顯示下採樣前後點數量（目標點雲）
        tgt_before = len(target_pcd.points)
        target_proc = preprocess_point_cloud(target_pcd, sample_step)
        tgt_after = len(target_proc.points)
        print(f"[Downsampling Target] {target_file}：{tgt_before} -> {tgt_after}")
        
        # 下採樣後若缺少法向量則重新估計
        if not source_proc.has_normals():
            source_proc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
        if not target_proc.has_normals():
            target_proc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
    else:
        source_proc = source_pcd
        target_proc = target_pcd

    # 使用 ICP 進行精細配準
    initial_transformation = np.eye(4)
    result_icp = refine_registration(source_proc, target_proc, initial_transformation, refined_distance_threshold)
    source_proc.transform(result_icp.transformation)
    
    merged_pcd = source_proc + target_proc
    # 著色為紅色 (RGB = [1, 0, 0])
    merged_pcd.paint_uniform_color([1, 0, 0])
    return merged_pcd

def get_new_filename(file1, file2):
    """
    檔案命名規則：  
    若 file1 為 "A_B.ply" 而 file2 為 "B_C.ply"，  
    則回傳 "A_C.ply"
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
    # 參數設定
    sample_step = 2                   # 下採樣取樣步長
    voxel_size = 0.1                  # 用於法向量估計及 ICP 門檻
    refined_distance_threshold = voxel_size * 1.5

    base_folder = r"C:/Users/ASUS/Desktop/POINT/red/furiren/processed/"
    
    # 初始化記錄列表，用來儲存「階段、檔名、點雲數量、耗時」資訊
    records = []
    
    # ------------------ 第一階段 ------------------
    # 原始檔案 normals_point_cloud_00000 ~ normals_point_cloud_00076
    output_folder = os.path.join(base_folder, "2")
    os.makedirs(output_folder, exist_ok=True)
    
    total_files = 77  # 原始檔案共77個
    # 第一階段全部皆採用下採樣
    for i in range(total_files - 1):
        source_file = os.path.join(base_folder, f"normals_point_cloud_{i:05d}.ply")
        target_file = os.path.join(base_folder, f"normals_point_cloud_{i+1:05d}.ply")
        if not os.path.exists(source_file) or not os.path.exists(target_file):
            print(f"檔案不存在，略過: {source_file} 或 {target_file}")
            continue
        
        print(f"第一階段配準: {source_file} 與 {target_file}")
        start_time = time.time()
        merged_pcd = process_pair(source_file, target_file, sample_step, voxel_size, refined_distance_threshold, downsample=True)
        end_time = time.time()
        merge_time = end_time - start_time
        output_filename = f"{i:05d}_{i+1:05d}.ply"
        output_path = os.path.join(output_folder, output_filename)
        o3d.io.write_point_cloud(output_path, merged_pcd)
        print(f"儲存至: {output_path}，耗時: {merge_time:.2f} 秒")
        # 記錄階段 2 的資訊（第一階段存放於 processed/2）
        records.append({
            "Stage": 2,
            "Filename": output_filename,
            "PointCount": len(merged_pcd.points),
            "MergeTime(s)": merge_time
        })
    
    # ------------------ 後續階段迭代 ------------------
    stage = 2  # 第一階段結果存於 processed/2
    # 在後續階段採用全域計數器：第一次以及每 3 次合併進行下採樣
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
            
            # 判斷是否進行下採樣：第一次合併或每 3 次合併時執行下採樣
            if merge_counter == 1 or merge_counter % 3 == 0:
                ds_flag = True
            else:
                ds_flag = False

            print(f"階段 {stage} 配準: {source_file} 與 {target_file} (downsample={ds_flag})")
            start_time = time.time()
            merged_pcd = process_pair(source_file, target_file, sample_step, voxel_size, refined_distance_threshold, downsample=ds_flag)
            end_time = time.time()
            merge_time = end_time - start_time
            new_filename = get_new_filename(file1, file2)
            output_path = os.path.join(next_folder, new_filename)
            o3d.io.write_point_cloud(output_path, merged_pcd)
            print(f"→ 輸出: {output_path}，耗時: {merge_time:.2f} 秒")
            # 記錄此階段的合併資訊，記錄所在階段為 next_stage
            records.append({
                "Stage": next_stage,
                "Filename": new_filename,
                "PointCount": len(merged_pcd.points),
                "MergeTime(s)": merge_time
            })
            merge_counter += 1  # 更新計數器
        
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
    
    # ------------------ 輸出 Excel 記錄 ------------------
    df = pd.DataFrame(records, columns=["Stage", "Filename", "PointCount", "MergeTime(s)"])
    excel_log_file = os.path.join(base_folder, "merge_log.xlsx")
    df.to_excel(excel_log_file, index=False)
    print("Merge log saved to:", excel_log_file)
