import open3d as o3d
import numpy as np

# 指定檔案路徑
source_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00000.ply"
target_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00010.ply"

# 加載點雲
source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 下採樣（可選）
voxel_size = 0.01
source_down = source.voxel_down_sample(voxel_size=voxel_size)
target_down = target.voxel_down_sample(voxel_size=voxel_size)

# 計算法向量
source_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
)
target_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
)

# 計算特徵
radius = 0.05
max_nn = 30
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
)
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
)

# RANSAC 初始對齊
max_correspondence_distance_init = 0.15
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, 
    target_down, 
    source_fpfh, 
    target_fpfh,
    mutual_filter=True,  # 添加 mutual_filter 參數
    max_correspondence_distance=max_correspondence_distance_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance_init)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
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
