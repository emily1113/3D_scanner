import pclpy
from pclpy import pcl
import numpy as np

# 讀取源點雲和目標點雲
source_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00000.ply"
target_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00001.ply"

# 讀取點雲
source_cloud = pcl.PointCloud.PointXYZ()
target_cloud = pcl.PointCloud.PointXYZ()

pcl.io.loadPLYFile(source_path, source_cloud)
pcl.io.loadPLYFile(target_path, target_cloud)

# NDT 配準設定
ndt = pcl.registration.NDT.PointXYZ()

# 設置網格體素大小
voxel_grid_size = 1.0  # 可根據點雲的尺度來調整
ndt.setResolution(voxel_grid_size)

# 設置最大迭代次數
ndt.setMaximumIterations(100)

# 設置配準步驟的變換容忍度（即收斂條件）
ndt.setTransformationEpsilon(1e-6)

# 設置匹配點間的最大距離（僅考慮此距離內的點進行匹配）
ndt.setStepSize(0.1)

# 設置源點雲和目標點雲
ndt.setInputSource(source_cloud)
ndt.setInputTarget(target_cloud)

# 初始變換矩陣（可選，如果有粗略的初始對齊）
initial_guess = np.identity(4, dtype=np.float32)

# 執行 NDT 配準
final_transform = pcl.registration.TransformationEstimationSVD.computeTransformation(
    ndt.align(source_cloud, initial_guess)
)

print("最終的 NDT 變換矩陣:")
print(final_transform)

# 使用最終變換矩陣對源點雲進行變換
source_transformed = pcl.PointCloud.PointXYZ()
pcl.registration.transformation.TransformationEstimationSVD.applyTransformation(
    source_cloud, final_transform, source_transformed
)

# 保存配準結果或可視化
pcl.io.savePLYFile("ndt_aligned.ply", source_transformed)
