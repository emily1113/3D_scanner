import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    """預處理點雲：下採樣和計算法線"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down


def compute_fpfh_feature(pcd, voxel_size):
    """計算點雲的 FPFH 特徵"""
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """基於特徵的粗略對齊 (RANSAC)"""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result


def execute_icp(source, target, trans_init, threshold):
    """執行精細 ICP 配準"""
    print("執行 ICP 配準...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    print("ICP 配準結果:")
    print(reg_p2p)
    print("最終變換矩陣:")
    print(reg_p2p.transformation)
    return reg_p2p.transformation


if __name__ == "__main__":
    # 讀取自定義點雲數據
    source_path = "C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00000.ply"
    target_path = "C:/Users/ASUS/Desktop/POINT/red/icp_5/point_cloud_00001.ply"
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # 設定參數
    voxel_size = 0.05  # 下採樣體素大小
    threshold = 0.02   # ICP 配準閾值

    # 預處理點雲
    print("預處理點雲...")
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    # 計算 FPFH 特徵
    print("計算 FPFH 特徵...")
    source_fpfh = compute_fpfh_feature(source_down, voxel_size)
    target_fpfh = compute_fpfh_feature(target_down, voxel_size)

    # 粗略對齊
    print("執行粗略對齊 (RANSAC)...")
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("RANSAC 粗略對齊結果:")
    print(result_ransac)
    print("粗略對齊變換矩陣:")
    print(result_ransac.transformation)

    # 繪製粗略對齊結果
    draw_registration_result(source, target, result_ransac.transformation)

    # 精細 ICP 配準
    print("執行精細 ICP 配準...")
    final_transformation = execute_icp(source, target, result_ransac.transformation, threshold)

    # 繪製最終配準結果
    draw_registration_result(source, target, final_transformation)
