import open3d as o3d
import numpy as np

# 讀取源點雲和目標點雲
source_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00000.ply"
target_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00001.ply"

source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 下採樣並計算 FPFH 特徵
def preprocess_point_cloud(pcd, voxel_size):
    print(":: 使用體素大小 %.3f 進行下採樣。" % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    print(":: 計算法向量。")
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    print(":: 計算 FPFH 特徵。")
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# 全局配準 (使用 FPFH 和 RANSAC)
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: 使用距離閾值 %.3f 進行 RANSAC 全局配準。" % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

# 局部精細配準 (使用 ICP)
def refine_registration(source, target, transformation, voxel_size):
    print(":: 執行 ICP 精細配準。")
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result

# 主函數執行全局特徵匹配和局部精細配準
voxel_size = 0.05  # 下採樣體素大小

# 預處理點雲，計算特徵
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# 執行全局配準
ransac_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print("RANSAC 配準變換矩陣:")
print(ransac_result.transformation)

# 使用全局配準結果作為初始變換，進行 ICP 精細配準
icp_result = refine_registration(source, target, ransac_result.transformation, voxel_size)
print("ICP 精細配準變換矩陣:")
print(icp_result.transformation)

# 可視化配準結果
print(":: 可視化配準結果。")
source.transform(icp_result.transformation)
o3d.visualization.draw_geometries([source, target], window_name="全局特徵匹配後的點雲配準結果")
