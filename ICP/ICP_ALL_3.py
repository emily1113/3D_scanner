import numpy as np
import open3d as o3d
import os

# ------------------ 讀取與下採樣 ------------------

def load_point_cloud(file_path):
    """
    使用 Open3D 讀取點雲檔案，回傳點雲物件
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def preprocess_point_cloud(point_cloud, sample_step):
    """
    均勻下採樣：利用 numpy 陣列切片 (每隔 sample_step 個點取一個)
    同時保留法向量與顏色資訊
    """
    points = np.asarray(point_cloud.points)
    downsampled_points = points[::sample_step]
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    
    # 如果有法向量就一起下採樣
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        downsampled_normals = normals[::sample_step]
        downsampled_pcd.normals = o3d.utility.Vector3dVector(downsampled_normals)
    
    # 如果有顏色就一起下採樣
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        downsampled_colors = colors[::sample_step]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)
    
    return downsampled_pcd

# ------------------ ICP 精細配準 ------------------

def refine_registration(source, target, initial_transformation, distance_threshold):
    """
    使用 ICP (Iterative Closest Point) 進行精細配準 (點到平面)
    """
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

# ------------------ 主程序 ------------------

if __name__ == "__main__":
    # 參數設定 (可依需要自行調整)
    sample_step = 2                # 均勻下採樣取樣步長
    voxel_size = 0.1               # 用於估計法向量及 ICP 門檻
    refined_distance_threshold = voxel_size * 1.5

    # 檔案所在資料夾，假設原始檔案都存在此資料夾中
    input_folder = r"C:/Users/ASUS/Desktop/POINT/red/furiren/processed/2"
    # 輸出結果的資料夾
    output_folder = r"C:/Users/ASUS/Desktop/POINT/red/furiren/processed/3"
    # 如果輸出資料夾不存在則自動建立
    os.makedirs(output_folder, exist_ok=True)

    # 設定要做幾組配準 (假設原始檔案從 00000_00001.ply 至 00075_00076.ply)
    # 此處i取自左側檔案編號，並以 (i, i+1) 與 (i+1, i+2) 配對
    start_idx = 0
    end_idx = 75  # 例如最後配對為 00075_00076 與 00076_00077, 若無 00076_00077，則依實際檔案調整

    for i in range(start_idx, end_idx + 1):
        # 定義來源檔名與目標檔名
        source_file = os.path.join(input_folder, f"{i:05d}_{i+1:05d}.ply")
        target_file = os.path.join(input_folder, f"{i+1:05d}_{i+2:05d}.ply")
        
        # 檢查檔案是否存在，若不存在則略過該組
        if not os.path.exists(source_file) or not os.path.exists(target_file):
            print(f"檔案不存在，略過: {source_file} 或 {target_file}")
            continue
        
        print(f"\n配準檔案: {source_file} 與 {target_file}")
        
        # 讀取點雲
        source_pcd = load_point_cloud(source_file)
        target_pcd = load_point_cloud(target_file)
        
        # 若原始點雲缺少法向量則估計
        if not source_pcd.has_normals():
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        if not target_pcd.has_normals():
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        # 均勻下採樣
        source_down = preprocess_point_cloud(source_pcd, sample_step)
        target_down = preprocess_point_cloud(target_pcd, sample_step)
        
        # 若下採樣後缺少法向量則重新估計
        if not source_down.has_normals():
            source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        if not target_down.has_normals():
            target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        # ICP 配準，使用單位矩陣作為初始變換
        initial_transformation = np.eye(4)
        result_icp = refine_registration(source_down, target_down, initial_transformation, refined_distance_threshold)
        
        print("ICP 對齊變換矩陣:")
        print(result_icp.transformation)
        
        # 套用變換至 source_down
        source_down.transform(result_icp.transformation)
        
        # 合併配準後的點雲 (來源與目標)
        merged_pcd = source_down + target_down
        
        # 新檔名：例如 (00000_00001 與 00001_00002) 配準後輸出檔名為 00000_00002.ply
        output_filename = f"{i:05d}_{i+2:05d}.ply"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"→ 輸出檔名: {output_filename}")
        o3d.io.write_point_cloud(output_path, merged_pcd)
