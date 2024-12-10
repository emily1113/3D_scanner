import open3d as o3d
import numpy as np

# 設定源和目標點雲的路徑
source_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00000.ply"
target_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply"

# 讀取點雲
source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 設置顏色以便於區分
source.paint_uniform_color([1, 0, 0])  # 紅色
target.paint_uniform_color([0, 1, 0])  # 綠色

# 手動設定初始變換矩陣
# 假設對源點雲進行一定的平移和旋轉來使其更接近目標點雲
rotation_angle = np.radians(5)  # 將角度轉換為弧度
y_rotation = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
                       [0, 1, 0, 0],
                       [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
                       [0, 0, 0, 1]])

x_translation = np.array([[1, 0, 0, -60],  # 在 x 軸負方向平移 60
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

trans_init = x_translation @ y_rotation

# 將初始變換應用到源點雲
source.transform(trans_init)

# 可視化初始對齊後的點雲
print("可視化手動初始對齊後的點雲")
o3d.visualization.draw_geometries([source, target], window_name="手動初始對齊")

# 設置 ICP 配準參數
threshold = 0.05  # 配準距離閾值

# 計算法向量
radius_normal = 0.1
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

# 使用點對平面的 ICP 進行精細配準
print("進行基於點對平面的 ICP 配準")
icp_result = o3d.pipelines.registration.registration_icp(
    source, target, threshold, np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

# 獲取 ICP 配準的變換矩陣
transformation = icp_result.transformation
print("ICP 變換矩陣:")
print(transformation)

# 將變換應用到源點雲（再次進行精細調整）
source.transform(transformation)

# 可視化配準後的點雲
print("可視化配準後的點雲")
o3d.visualization.draw_geometries([source, target], window_name="ICP 配準結果")
