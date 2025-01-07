import open3d as o3d
import numpy as np

def compute_pfh(point_cloud, radius):
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    pfh_features = []
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    for i in range(len(points)):
        [_, idx, _] = kdtree.search_radius_vector_3d(points[i], radius)
        if len(idx) < 5:
            pfh_features.append(np.zeros(33))
            continue

        hist = np.zeros(33)
        for j in idx[1:]:
            u = normals[i]
            v = normals[j]
            diff = points[j] - points[i]

            d = np.linalg.norm(diff)
            alpha = np.arctan2(np.dot(np.cross(u, diff), v), np.dot(u, diff))
            phi = np.dot(u, diff) / d
            theta = np.dot(v, diff) / d

            alpha_idx = int((alpha + np.pi) / (2 * np.pi / 11))
            phi_idx = int((phi + 1) / (2 / 11))
            theta_idx = int((theta + 1) / (2 / 11))
            hist[alpha_idx] += 1
            hist[phi_idx + 11] += 1
            hist[theta_idx + 22] += 1

        pfh_features.append(hist / len(idx))

    return np.array(pfh_features)

# 加載點雲
source_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00000.ply"
target_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00010.ply"

source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 下採樣
voxel_size = 0.05
source_down = source.voxel_down_sample(voxel_size=voxel_size)
target_down = target.voxel_down_sample(voxel_size=voxel_size)

# 設定 PFH 計算參數
radius_feature = 0.1

# 計算 PFH 特徵
source_pfh = compute_pfh(source_down, radius_feature)
target_pfh = compute_pfh(target_down, radius_feature)

# 創建 Open3D Feature 對象
source_feature = o3d.pipelines.registration.Feature()
target_feature = o3d.pipelines.registration.Feature()
source_feature.data = source_pfh.T  # 注意需要轉置
target_feature.data = target_pfh.T  # 注意需要轉置

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
