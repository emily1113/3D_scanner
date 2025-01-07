import open3d as o3d
import numpy as np

# 載入點雲
pcd = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00003.ply")

# 下採樣以提升效率
pcd_down = pcd.voxel_down_sample(voxel_size=0.05)

# 計算法向量
pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 計算FPFH特徵
radius_feature = 0.25  # 搜索半徑
fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
)

# 隨機選取一部分點作為特徵點
num_points = len(np.asarray(pcd_down.points))
num_features_to_highlight = 50  # 顯示的特徵點數量
selected_indices = np.random.choice(num_points, num_features_to_highlight, replace=False)

# 創建特徵點雲
feature_points = o3d.geometry.PointCloud()
feature_points.points = o3d.utility.Vector3dVector(np.asarray(pcd_down.points)[selected_indices])
feature_points.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (num_features_to_highlight, 1)))  # 紅色特徵點

# 設置原始點雲顏色
pcd_down.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

# 顯示點雲和特徵點
o3d.visualization.draw_geometries([pcd_down, feature_points],
                                  window_name="Feature Points",
                                  point_show_normal=False)
