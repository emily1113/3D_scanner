import os
import re
import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    """
    從指定路徑讀取 PLY 格式的點雲文件，並將點雲轉換為 NumPy 陣列。
    
    參數:
        file_path: str
            檔案的完整路徑

    返回:
        numpy.ndarray: 讀取的點雲數據，每行代表一個 (x, y, z) 坐標
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        point_cloud = np.asarray(pcd.points)
        # 檢查資料是否至少有 x, y, z 三個維度
        if point_cloud.shape[1] < 3:
            raise ValueError("點雲文件必須包含至少 x, y, z 三個數據！")
        return point_cloud
    except Exception as e:
        print(f"讀取檔案 {file_path} 失敗:", e)
        return None

def voxel_downsample(point_cloud, voxel_size):
    """
    體素下採樣：利用指定的體素大小對點雲進行下採樣處理，
    將同一體素中的多個點替換成該體素所有點的重心。

    參數:
        point_cloud: numpy.ndarray, shape (N, 3)
            原始點雲數據，每行代表一個點 (x, y, z)
        voxel_size: float
            體素邊長（下採樣解析度）

    返回:
        numpy.ndarray: 下採樣後的點雲
    """
    # 計算原始點雲的最小邊界，使所有點移動後索引從 0 開始
    min_bounds = np.min(point_cloud, axis=0)
    
    # 為每個點計算其所在體素的索引
    voxel_indices = np.floor((point_cloud - min_bounds) / voxel_size).astype(np.int32)
    
    # 用字典彙集同一體素中的點
    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        voxel_dict.setdefault(key, []).append(point_cloud[i])
    
    # 計算每個體素中所有點的重心作為代表點
    downsampled_points = []
    for points in voxel_dict.values():
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        downsampled_points.append(centroid)
    
    return np.array(downsampled_points)

if __name__ == '__main__':
    # 指定要讀取的資料夾路徑，請根據實際情況修改
    folder_path = r"C:\Users\ASUS\Desktop\POINT\red\furiren\processed"
    # 指定輸出資料夾，用於存放下採樣後的檔案
    output_folder = os.path.join(folder_path, "downsampled")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 設定體素下採樣的體素大小
    voxel_size = 0.5  # 可根據需要進行調整

    # 遍歷資料夾中所有的 .ply 檔案
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".ply"):
            input_file = os.path.join(folder_path, file_name)
            print(f"正在處理檔案: {input_file}")
            
            point_cloud = load_point_cloud(input_file)
            if point_cloud is None:
                print("讀取失敗，跳過此檔案。")
                continue
            
            print(f"成功讀取點雲，點數 = {point_cloud.shape[0]}")
            downsampled_cloud = voxel_downsample(point_cloud, voxel_size)
            print(f"下採樣後點數 = {downsampled_cloud.shape[0]}")
            
            # 轉換回 Open3D 點雲物件
            down_pcd = o3d.geometry.PointCloud()
            down_pcd.points = o3d.utility.Vector3dVector(downsampled_cloud)
            down_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            
            # 提取輸入檔案中數字部分作為輸出檔名
            match = re.search(r"(\d+)", file_name)
            if match:
                numeric_part = match.group(1)
            else:
                numeric_part = os.path.splitext(file_name)[0]
            
            # 設定輸出檔案名稱，例如 "12345.ply"
            output_file = os.path.join(output_folder, f"{numeric_part}.ply")
            o3d.io.write_point_cloud(output_file, down_pcd)
            print(f"已儲存下採樣後的點雲至: {output_file}\n")
