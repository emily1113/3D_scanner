import open3d as o3d
import numpy as np

# 假設我們已經實現了 SHOT 特徵計算的 `compute_shot` 函數
def compute_shot(point_cloud, radius):
    # 簡化版，僅作示例
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    shot_features = []
    for i in range(len(points)):
        [_, idx, _] = kdtree.search_radius_vector_3d(points[i], radius)
        if len(idx) < 5:
            shot_features.append(np.zeros(352))  # 假設 SHOT 特徵為 352 維
            continue

        # 特徵計算邏輯（此處為簡化示例）
        hist = np.zeros(352)  # 假設分箱數為 352
        shot_features.append(hist)

    return np.array(shot_features)

# 加載點雲
source_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00000.ply"
target_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00005.ply"

source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 下採樣
voxel_size = 0.05
source_down = source.voxel_down_sample(voxel_size=voxel_size)
target_down = target.voxel_down_sample(voxel_size=voxel_size)

# 計算法向量
source_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
)
target_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
)

# 設定 SHOT 特徵計算參數
radius_feature = 0.1
source_shot = compute_shot(source_down, radius_feature)
target_shot = compute_shot(target_down, radius_feature)

# 創建 Open3D Feature 對象
source_feature = o3d.pipelines.registration.Feature()
target_feature = o3d.pipelines.registration.Feature()
source_feature.data = source_shot.T
target_feature.data = target_shot.T

# RANSAC 初始對齊
max_correspondence_distance = 0.15
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down,
    source_feature, target_feature,
    mutual_filter=True,
    max_correspondence_distance=max_correspondence_distance,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
)

print("RANSAC transformation:")
print(ransac_result.transformation)

# ICP 精細對齊
max_correspondence_distance_icp = 0.05
icp_result = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance=max_correspondence_distance_icp,
    init=ransac_result.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("ICP transformation:")
print(icp_result.transformation)

# 可視化結果
source.transform(icp_result.transformation)
o3d.visualization.draw_geometries([source, target])
