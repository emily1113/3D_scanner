import open3d as o3d
import numpy as np

# 读取参考点云和目标点云
source = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00000.ply")  # 目标点云
target = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00001.ply")  # 参考点云

# 对点云进行下采样以减少计算量
source_down = source.voxel_down_sample(voxel_size=0.05)
target_down = target.voxel_down_sample(voxel_size=0.05)

# 计算法向量
source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 配准的初始变换
initial_transformation = np.eye(4)

# 使用点到点ICP进行点云配准
icp_result = o3d.pipelines.registration.registration_icp(
    source_down, target_down, max_correspondence_distance=0.1,
    init=initial_transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

# 打印结果
print("ICP 结果：")
print(icp_result)
print("优化后的变换矩阵：")
print(icp_result.transformation)

# 应用优化后的变换
source_down.transform(icp_result.transformation)

# 可视化对齐后的点云
o3d.visualization.draw_geometries([source_down, target_down])
