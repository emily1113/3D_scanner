import os
import numpy as np
import open3d as o3d

# ------------------ 讀取點雲 ------------------
def load_point_cloud(file_path):
    """
    使用 Open3D 讀取點雲檔案，回傳點雲物件
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# ------------------ 下採樣 ------------------
def preprocess_point_cloud(point_cloud, sample_step):
    """
    均勻下採樣：利用 numpy 陣列切片 (每隔 sample_step 個點取一個)
    同時保留法向量與顏色資訊
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

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 指定資料夾路徑，並取得前 5 個 .ply 檔案
    folder_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/processed"
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".ply")]
    file_list.sort()  # 排序以確保順序
    file_list = file_list[:5]  # 僅處理前 5 個檔案
    
    # 參數設定
    sample_step = 2          # 下採樣步長
    voxel_size = 0.1         # 用於法向量估計與 ICP 門檻的體素尺寸
    refined_distance_threshold = voxel_size * 1.5  # ICP 精細配準門檻

    # 讀取第一個點雲作為初始累積點雲
    print("讀取初始點雲:", file_list[0])
    accumulated_pcd = load_point_cloud(file_list[0])
    if not accumulated_pcd.has_normals():
        accumulated_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    accumulated_down = preprocess_point_cloud(accumulated_pcd, sample_step)
    # 初始累積點雲著紅色
    accumulated_down.paint_uniform_color([1, 0, 0])
    
    # 顯示初始結果
    display_point_clouds(accumulated_down, title="累積結果：檔案 1")
    
    # 從第 2 個檔案開始依序進行配準
    for idx, file in enumerate(file_list[1:], start=2):
        print(f"\n處理檔案: {file}")
        current_pcd = load_point_cloud(file)
        if not current_pcd.has_normals():
            current_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
        current_down = preprocess_point_cloud(current_pcd, sample_step)
        # 當前點雲著綠色
        current_down.paint_uniform_color([0, 1, 0])
        
        # 顯示匹配前的狀態：累積點雲與待匹配點雲同時展示
        display_point_clouds(accumulated_down, current_down, title=f"匹配前狀態：檔案 {idx}")
        
        # 使用 ICP 將 current_down 與累積點雲 accumulated_down 配準
        initial_transformation = np.eye(4)
        result_icp = refine_registration(current_down, accumulated_down, initial_transformation, refined_distance_threshold)
        print("ICP 精細配準結果:")
        print(result_icp)
        print("變換矩陣:")
        print(result_icp.transformation)
        
        # 將 ICP 變換結果應用於 current_down，使其與累積點雲對齊
        current_down.transform(result_icp.transformation)
        
        # 合併配準後的點雲：累積點雲與當前配準結果
        accumulated_points = np.vstack((np.asarray(accumulated_down.points), np.asarray(current_down.points)))
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(accumulated_points)
        
        # 合併顏色資訊（若有）
        if accumulated_down.has_colors() and current_down.has_colors():
            accumulated_colors = np.vstack((np.asarray(accumulated_down.colors), np.asarray(current_down.colors)))
            merged_pcd.colors = o3d.utility.Vector3dVector(accumulated_colors)
        
        # 合併法向量資訊（若有）
        if accumulated_down.has_normals() and current_down.has_normals():
            accumulated_normals = np.vstack((np.asarray(accumulated_down.normals), np.asarray(current_down.normals)))
            merged_pcd.normals = o3d.utility.Vector3dVector(accumulated_normals)
        
        # 重新估計合併後的法向量
        merged_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        
        # 更新累積結果
        accumulated_down = merged_pcd
        
        # 顯示匹配後的累積結果
        display_point_clouds(accumulated_down, title=f"累積結果：檔案 {idx}")
