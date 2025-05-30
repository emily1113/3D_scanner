import numpy as np
import open3d as o3d
import copy  # 用於深度拷貝

# 1. 加載內建的 Stanford Bunny 點雲
def load_bunny_data():
    bunny_source = o3d.data.BunnyMesh().path  # 內建 Stanford Bunny 模型
    mesh = o3d.io.read_triangle_mesh(bunny_source)
    mesh.compute_vertex_normals()
    point_cloud = mesh.sample_points_poisson_disk(1000)  # 從網格生成點雲
    return point_cloud

# 2. 加載點雲並降採樣
def preprocess_point_cloud(point_cloud, voxel_size):
    point_cloud_down = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    point_cloud_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return point_cloud_down

# 3. 計算 FPFH 特徵
def compute_fpfh(point_cloud, voxel_size):
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return fpfh

# 4. 使用 SAC-IA 進行初始對齊
def sac_ia_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
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

# 5. 模擬更大的平移和旋轉
def apply_random_transform(point_cloud):
    """
    對點雲應用隨機變換，包括旋轉和平移。
    """
    # 隨機旋轉矩陣
    theta = np.radians(45)  # 旋轉 45 度
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    # 平移向量
    translation = np.array([0.5, 0.5, 0.2])  # 更大的平移
    # 構造 4x4 變換矩陣
    transform_matrix = np.vstack((np.hstack((rotation_matrix, translation[:, None])), [0, 0, 0, 1]))
    point_cloud.transform(transform_matrix)
    return point_cloud

# 主程序
if __name__ == "__main__":
    voxel_size = 0.01  # 體素大小

    # 加載 Bunny 點雲
    source = load_bunny_data()
    target = copy.deepcopy(source)  # 使用 deepcopy 替代 clone
    target = apply_random_transform(target)  # 對目標點雲應用隨機變換

    # 降採樣和法向量估算
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    # 計算 FPFH 特徵
    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    # 可視化擬合前的狀態
    print("展示擬合前的點雲...")
    o3d.visualization.draw_geometries([source_down.paint_uniform_color([1, 0, 0]),  # 紅色
                                       target_down.paint_uniform_color([0, 1, 0])])  # 綠色

    # 使用 SAC-IA 進行初始對齊
    result = sac_ia_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # 輸出對齊結果
    print("SAC-IA 初始對齊結果:")
    print(result)
    print("對齊變換矩陣:")
    print(result.transformation)

    # 可視化擬合後的狀態
    print("展示擬合後的點雲...")
    source_down.transform(result.transformation)
    o3d.visualization.draw_geometries([source_down.paint_uniform_color([1, 0, 0]),  # 紅色
                                       target_down.paint_uniform_color([0, 1, 0])])  # 綠色
    
