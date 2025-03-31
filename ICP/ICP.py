import numpy as np
import open3d as o3d
import re

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

    參數:
      point_cloud : 原始點雲（Open3D 格式）
      sample_step : 下採樣步長，數值越小保留的點越多

    回傳:
      downsampled_pcd : 下採樣後的點雲（Open3D 格式）
    """
    original_point_count = len(point_cloud.points)
    print(f"原始點雲點數量: {original_point_count}")
    
    points = np.asarray(point_cloud.points)
    sampled_points = points[::sample_step]
    
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        sampled_normals = normals[::sample_step]
        downsampled_pcd.normals = o3d.utility.Vector3dVector(sampled_normals)
    
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        sampled_colors = colors[::sample_step]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)
    
    down_sampled_point_count = len(downsampled_pcd.points)
    print(f"下採樣後點雲點數量: {down_sampled_point_count}")
    
    return downsampled_pcd

# ------------------ ICP 精細配準 ------------------

def refine_registration(source, target, initial_transformation, distance_threshold):
    """
    使用 ICP (Iterative Closest Point) 進行精細配準
    此處採用點到平面方法，使用 Open3D 內建的 ICP 函式

    參數:
      source : 源點雲（下採樣後）
      target : 目標點雲（下採樣後）
      initial_transformation : 初始變換矩陣（通常為單位矩陣）
      distance_threshold : 對應點距離門檻

    回傳:
      result_icp : ICP 配準結果物件，包含最終變換矩陣等資訊
    """
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=distance_threshold,
        init=initial_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

# ------------------ 可視化 ------------------

def display_point_clouds(*point_clouds, title="點雲展示"):
    """
    使用 Open3D 視覺化點雲
    """
    o3d.visualization.draw_geometries(list(point_clouds), window_name=title)

def extract_number(file_path):
    """
    從檔案路徑中擷取檔名中的數字部分
    假設檔名格式為 ..._<數字>.ply
    """
    filename = file_path.split("/")[-1]  # 取得檔名（若使用 Windows，可能需要用 '\\' 分隔）
    match = re.search(r'_(\d+)\.ply$', filename)
    if match:
        return match.group(1)
    else:
        return "unknown"

# ------------------ 主程序 ------------------

if __name__ == "__main__":
    # 參數設定
    sample_step = 2          # 每隔 2 個點取一個進行下採樣
    voxel_size = 0.1         # 用於設定 ICP 配準的門檻
    distance_threshold = voxel_size * 3.0   # ICP 內點門檻
    refined_distance_threshold = voxel_size * 1.5  # ICP 精細配準門檻

    # 指定 source 與 target 點雲檔案路徑
    source_file = "C:/Users/ASUS/Desktop/POINT/red/furiren/processed/normals_point_cloud_00000.ply"
    target_file = "C:/Users/ASUS/Desktop/POINT/red/furiren/processed/normals_point_cloud_00004.ply"

    # 讀取點雲
    source = load_point_cloud(source_file)
    target = load_point_cloud(target_file)

    # 若原始點雲缺少法向量則估計法向量
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 均勻下採樣
    source_down = preprocess_point_cloud(source, sample_step)
    target_down = preprocess_point_cloud(target, sample_step)

    # 若下採樣後缺少法向量則重新估計
    if not source_down.has_normals():
        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if not target_down.has_normals():
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 配準前視覺化，將 source 著紅色，target 著綠色
    source_down.paint_uniform_color([1, 0, 0])
    target_down.paint_uniform_color([0, 1, 0])
    display_point_clouds(source_down, target_down, title="下採樣點雲展示")

    # 初始對齊，這裡使用單位矩陣作為初始變換
    initial_transformation = np.eye(4)

    # 使用 ICP 進行精細配準
    result_icp = refine_registration(source_down, target_down, initial_transformation, refined_distance_threshold)
    print("ICP 精細配準結果:")
    print(result_icp)
    print("ICP 對齊變換矩陣:")
    print(result_icp.transformation)

    # 將 ICP 配準結果應用於 source 點雲並視覺化
    source_down.transform(result_icp.transformation)
    source_down.paint_uniform_color([0, 0, 1])
    display_point_clouds(source_down, target_down, title="ICP 配準結果")

    # ------------------ 儲存匹配結果 ------------------
    # 合併配準後的 source 與 target 點雲
    merged_pcd = source_down + target_down

    # 從原始檔案名稱中提取數字編號
    source_num = extract_number(source_file)
    target_num = extract_number(target_file)
    output_filename = f"{source_num}_{target_num}.ply"
    print(f"儲存檔名: {output_filename}")

    # 儲存合併後的點雲
    o3d.io.write_point_cloud(output_filename, merged_pcd)
