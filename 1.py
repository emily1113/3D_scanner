import open3d as o3d
import numpy as np

# 加載 DemoColoredICPPointClouds 示例數據
demo_icp_data = o3d.data.DemoColoredICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_data.paths[0])  # 第一個點雲作為源點雲
o3d.visualization.draw_geometries([source])
target = o3d.io.read_point_cloud(demo_icp_data.paths[1])  # 第二個點雲作為目標點雲
o3d.visualization.draw_geometries([target])

# 可視化初始的源點雲和目標點雲
print("可視化初始的源點雲和目標點雲")
o3d.visualization.draw_geometries([source, target])

# 設定彩色 ICP 的參數
voxel_radius = [0.05, 0.03, 0.01]  # 不同精度的體素大小
max_iter = [80, 50, 40]  # 對應於每個體素大小的最大迭代次數

# 初始對齊 (使用普通 ICP 進行粗略對齊)
print("執行初始點雲對齊")
trans_init = np.identity(4)
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, 0.01, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
source.transform(reg_p2p.transformation)


print("執行彩色 ICP 配準")
current_transformation = reg_p2p.transformation

for scale in range(len(voxel_radius)):
    # 對源點雲和目標點雲進行降採樣
    iter_num = max_iter[scale]
    radius = voxel_radius[scale]
    
    print(f"在體素大小為 {radius} 下進行配準，最大迭代次數為 {iter_num}")
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)
    
    # 計算法線
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    
    # 彩色 ICP 對齊
    reg_icp = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, radius, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter_num)
    )
    
    # 更新變換矩陣
    current_transformation = reg_icp.transformation
    source.transform(reg_icp.transformation)

# 可視化最終對齊結果
print("彩色 ICP 配準完成，顯示最終對齊結果")
o3d.visualization.draw_geometries([source, target])
