import numpy as np
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors

def load_point_cloud(file_path):
    """
    讀取指定路徑的點雲檔案，並返回 open3d.geometry.PointCloud 物件。
    
    參數:
        file_path (str): 點雲檔案的完整路徑。
        
    回傳:
        open3d.geometry.PointCloud: 讀取的點雲。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"無法讀取點雲檔案: {file_path}")
    return pcd

def compute_normals_custom(points, k=15):
    """
    利用 k 近鄰與 PCA 計算點雲每個點的法向量。
    
    參數:
        points (np.ndarray): (N, 3) 的點雲座標陣列。
        k (int): 鄰域中使用的點數量。
    
    回傳:
        np.ndarray: (N, 3) 的法向量陣列。
    """
    n_points = points.shape[0]
    normals = np.zeros_like(points)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        neighbor_pts = points[indices[i]]
        mean = neighbor_pts.mean(axis=0)
        cov = np.dot((neighbor_pts - mean).T, (neighbor_pts - mean)) / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 取最小特徵值所對應的特徵向量作為法向量
        normals[i] = eigenvectors[:, 0]
    return normals

def preprocess_point_cloud(point_cloud, voxel_size):
    """
    對點雲進行體素降採樣並利用自訂方法估計法向量。
    
    參數:
        point_cloud (open3d.geometry.PointCloud): 原始點雲。
        voxel_size (float): 降採樣的體素大小。
        
    回傳:
        open3d.geometry.PointCloud: 降採樣且估計好法線的點雲。
    """
    # 體素降採樣
    pcd_down = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    # 利用自訂方法計算法向量
    points_down = np.asarray(pcd_down.points)
    normals = compute_normals_custom(points_down, k=15)
    pcd_down.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd_down

def compute_fpfh(point_cloud, voxel_size):
    """
    計算點雲的 FPFH 特徵。
    
    參數:
        point_cloud (open3d.geometry.PointCloud): 降採樣且估計好法線的點雲。
        voxel_size (float): 體素大小，用於設置搜索半徑。
        
    回傳:
        open3d.pipelines.registration.Feature: FPFH 特徵。
    """
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return fpfh

def sac_ia_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    """
    使用 SAC-IA（Sample Consensus Initial Alignment）進行點雲初始對齊。
    
    參數:
        source (open3d.geometry.PointCloud): 降採樣後的源點雲。
        target (open3d.geometry.PointCloud): 降採樣後的目標點雲。
        source_fpfh (open3d.pipelines.registration.Feature): 源點雲的 FPFH 特徵。
        target_fpfh (open3d.pipelines.registration.Feature): 目標點雲的 FPFH 特徵。
        voxel_size (float): 體素大小，用於設定對應的距離閾值。
        
    回傳:
        registration result: 包含初始對齊結果及變換矩陣。
    """
    distance_threshold = voxel_size * 3
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def display_source_and_target(source, target):
    """
    顯示初始的 source（紅色）與 target（綠色）點雲。
    """
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.paint_uniform_color([1, 0, 0])  # 紅色
    target_copy.paint_uniform_color([0, 1, 0])  # 綠色
    o3d.visualization.draw_geometries(
        [source_copy, target_copy],
        window_name="Initial Source (Red) and Target (Green) Point Clouds"
    )

def display_alignment_result(source_aligned, target):
    """
    顯示對齊後的點雲結果（source 為紅色，target 為綠色）。
    """
    source_copy = copy.deepcopy(source_aligned)
    target_copy = copy.deepcopy(target)
    source_copy.paint_uniform_color([1, 0, 0])
    target_copy.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries(
        [source_copy, target_copy],
        window_name="Aligned Point Clouds"
    )

if __name__ == "__main__":
    # 參數設定
    voxel_size = 0.05  # 體素大小，可根據點雲尺度進行調整

    # 指定 source 與 target 的點雲檔案路徑（請依實際情況修改）
    source_file = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00001.ply"
    target_file = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00010.ply"

    # 讀取點雲
    print("讀取點雲資料...")
    source = load_point_cloud(source_file)
    target = load_point_cloud(target_file)

    # 顯示初始點雲
    print("展示初始的點雲...")
    display_source_and_target(source, target)

    # 降採樣與自訂法向量估算
    print("進行點雲預處理 (降採樣與自訂法向量估算)...")
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    # 計算 FPFH 特徵
    print("計算 FPFH 特徵...")
    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # 使用 SAC-IA 進行初始對齊
    print("進行 SAC-IA 初始對齊...")
    result = sac_ia_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # 輸出對齊結果
    print("SAC-IA 初始對齊結果:")
    print(result)
    print("對齊變換矩陣:")
    print(result.transformation)

    # 將對齊變換應用到 source 點雲上
    source_aligned = copy.deepcopy(source_down)
    source_aligned.transform(result.transformation)

    # 顯示對齊後的點雲
    print("展示對齊後的點雲...")
    display_alignment_result(source_aligned, target_down)