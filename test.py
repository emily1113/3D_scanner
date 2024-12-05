import open3d as o3d
import numpy as np

# 設定源和目標點雲的路徑
source_path = "C:/Users/ASUS/Desktop/point_cloud_00000 - Cloud.ply"
target_path = "C:/Users/ASUS/Desktop/point_cloud_00001 - Cloud.ply"

# 讀取點雲
source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 設置顏色以便於區分
source.paint_uniform_color([1, 0, 0])  # 紅色
target.paint_uniform_color([0, 1, 0])  # 綠色

# 下採樣點雲以減少計算量並提取顯著特徵
voxel_size = 0.05  # 設定下採樣體素大小
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)

# 計算法向量
radius_normal = 0.1
source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
    radius=radius_normal, max_nn=30))
target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
    radius=radius_normal, max_nn=30))

# 使用點對平面的 ICP 進行配準
threshold = 0.01  # 配準距離閾值
trans_init = np.identity(4)  # 初始變換矩陣

print("進行基於點對平面的 ICP 配準")
icp_result = o3d.pipelines.registration.registration_icp(
    source_down, target_down, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

# 獲取 ICP 配準的變換矩陣
transformation = icp_result.transformation
print("ICP 變換矩陣:")
print(transformation)

# 將變換應用到源點雲（非下採樣的原始點雲）
source.transform(transformation)

# 可視化配準後的點雲
print("可視化配準後的點雲")
o3d.visualization.draw_geometries([source, target], window_name="ICP 配準結果")
