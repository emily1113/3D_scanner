import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy  # 用於深度拷貝

# 1. 加載內建的 Stanford Bunny 點雲
def load_bunny_data():
    bunny_source = o3d.data.BunnyMesh().path  # 內建 Stanford Bunny 模型
    mesh = o3d.io.read_triangle_mesh(bunny_source)
    mesh.compute_vertex_normals()
    point_cloud = mesh.sample_points_poisson_disk(2000)  # 從網格生成點雲
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
# 5. 對點雲應用隨機變換
def apply_random_transform(point_cloud):
    theta = np.radians(45)  # 旋轉 45 度
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    translation = np.array([0.5, 0.5, 0.2])  # 更大的平移
    transform_matrix = np.vstack((np.hstack((rotation_matrix, translation[:, None])), [0, 0, 0, 1]))
    
    # 顯示齊次變換矩陣
    print("生成的齊次變換矩陣:")
    print(transform_matrix)
    
    # 將變換應用到點雲
    point_cloud.transform(transform_matrix)
    return point_cloud, transform_matrix


# 6. 儲存點雲為 PLY 並顯示
def save_and_display_point_cloud(point_cloud, filename, color=None):
    if color:
        point_cloud.paint_uniform_color(color)
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"點雲已儲存為 PLY 檔案: {filename}")
    o3d.visualization.draw_geometries([point_cloud], window_name=f"Point Cloud: {filename}")

# 7. 比較兩個旋轉矩陣是否一致
def compare_rotation_matrices(original_matrix, estimated_matrix):
    # 提取旋轉矩陣
    original_rotation = original_matrix[:3, :3]
    estimated_rotation = estimated_matrix[:3, :3]

    # 計算 Frobenius norm 差異
    diff = np.linalg.norm(original_rotation - estimated_rotation)
    print(f"旋轉矩陣差異 (Frobenius norm): {diff}")

    # 計算相對旋轉角度
    relative_rotation = np.dot(original_rotation.T, estimated_rotation)
    trace = np.trace(relative_rotation)
    angle = np.arccos((trace - 1) / 2)
    angle_degrees = np.degrees(angle)
    print(f"相對旋轉角度 (degrees): {angle_degrees}")

    # 判斷是否一致
    if diff < 1e-6 and angle_degrees < 1e-3:
        print("計算出的旋轉矩陣與原先變換矩陣一致。")
    else:
        print("計算出的旋轉矩陣與原先變換矩陣不一致。")

# 8. 同時展示初始的 source 和 target 點雲
def display_source_and_target(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])  # 紅色
    target_temp.paint_uniform_color([0, 1, 0])  # 綠色
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        window_name="Initial Source and Target Point Clouds",
        width=800,
        height=600,
        point_show_normal=False,
    )

# 9. 同時展示初始對齊結果與目標點雲
def display_alignment_result(source, target, aligned_source):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    aligned_source_temp = copy.deepcopy(aligned_source)

    # 設置顏色
    source_temp.paint_uniform_color([1, 0, 0])  # 紅色
    target_temp.paint_uniform_color([0, 1, 0])  # 綠色
    aligned_source_temp.paint_uniform_color([0, 0, 1])  # 藍色

    o3d.visualization.draw_geometries(
        [source_temp, target_temp, aligned_source_temp],
        window_name="Source, Target, and Aligned Result",
        width=800,
        height=600,
        point_show_normal=False,
    )

# 主程序
if __name__ == "__main__":
    voxel_size = 0.01  # 體素大小

    # 加載 Bunny 點雲
    source = load_bunny_data()
    target = copy.deepcopy(source)  # 使用 deepcopy 替代 clone
    target, original_transform = apply_random_transform(target)  # 對目標點雲應用隨機變換，並保存變換矩陣

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
    # 比較旋轉矩陣是否一致
    compare_rotation_matrices(original_transform, result.transformation)
    


# # 主程式
# if __name__ == "__main__":
#     voxel_size = 0.01  # 體素大小

#     # 加載 Bunny 點雲
#     source = load_bunny_data()
#     target = copy.deepcopy(source)
#     target, original_transform = apply_random_transform(target)  # 對目標點雲應用隨機變換，並保存變換矩陣

#     # 新增：展示初始的 source 和 target 點雲
#     display_source_and_target(source, target)

#     # 降採樣和法向量估算
#     source_down = preprocess_point_cloud(source, voxel_size)
#     target_down = preprocess_point_cloud(target, voxel_size)

#     # 儲存並顯示降採樣後的源和目標點雲
#     save_and_display_point_cloud(source_down, "source_down.ply", color=[1, 0, 0])  # 紅色
#     save_and_display_point_cloud(target_down, "target_down.ply", color=[0, 1, 0])  # 綠色

#     # 計算 FPFH 特徵
#     source_fpfh = compute_fpfh(source_down, voxel_size)
#     target_fpfh = compute_fpfh(target_down, voxel_size)

#     # 使用 SAC-IA 進行初始對齊
#     result = sac_ia_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

#     # 輸出對齊結果
#     print("SAC-IA 初始對齊結果:")
#     print(result)
#     print("對齊變換矩陣:")
#     print(result.transformation)

#     # 比較旋轉矩陣是否一致
#     compare_rotation_matrices(original_transform, result.transformation)

#     # 將變換應用到源點雲（SAC-IA 初始對齊結果）
#     source_aligned = copy.deepcopy(source_down)
#     source_aligned.transform(result.transformation)

#     # 儲存並顯示初始對齊結果
#     save_and_display_point_cloud(source_aligned, "sac_ia_aligned.ply", color=[1, 0, 0])  # 紅色
#     save_and_display_point_cloud(target_down, "target.ply", color=[0, 1, 0])  # 綠色

#     # 新增：同時展示初始的源點雲、目標點雲以及對齊結果
#     display_alignment_result(source_down, target_down, source_aligned)
