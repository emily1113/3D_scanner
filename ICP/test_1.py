import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation, zoom=0.7):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=zoom,
                                      front=[0.0, -0.0, -1.0],
                                      lookat=[41.587005615234375, -45.81600570678711, 743.3211212473857],
                                      up=[0.0, -1.0, -0.0])

def icp_registration(source_path, target_path):
    # 讀取點雲
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # 下採樣點雲以提高ICP效率
    voxel_size = 0.02  # 設置下採樣的體素大小
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # 計算法線
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 檢查法線是否已計算成功
    if not source_down.has_normals() or not target_down.has_normals():
        raise RuntimeError("法線計算失敗，請檢查參數設定")

    # 初始配準 (粗配準)
    threshold = 0.05
    trans_init = np.identity(4)
    print("粗配準開始...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print("粗配準結果:")
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation, zoom=0.7)

    # 精細配準之前，為完整的來源和目標點雲計算法線
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 設定法線的方向一致
    source.orient_normals_consistent_tangent_plane(k=10)
    target.orient_normals_consistent_tangent_plane(k=10)

    # 再次檢查法線是否已計算成功
    if not source.has_normals() or not target.has_normals():
        raise RuntimeError("法線計算失敗，請檢查參數設定")

    # 精細配準
    print("ICP精細配準開始...")
    threshold = 0.02  # ICP配準容差
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_icp.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print("精細配準結果:")
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation, zoom=0.7)

if __name__ == "__main__":
    source_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_1/point_cloud_00000.ply"  # 替換為來源點雲檔案的路徑
    target_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_1/point_cloud_00001.ply"  # 替換為目標點雲檔案的路徑
    icp_registration(source_path, target_path)
