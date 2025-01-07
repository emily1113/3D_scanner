import open3d as o3d
import numpy as np

# 設置源和目標點雲的檔案路徑
source_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply"
target_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00003.ply"

# 載入點雲
source = o3d.io.read_point_cloud(source_path)
target = o3d.io.read_point_cloud(target_path)

# 計算法向量
def estimate_normals(point_cloud, radius=0.1, max_nn=30):
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

estimate_normals(source)
estimate_normals(target)

# 計算 FPFH 特徵
def compute_fpfh(point_cloud, radius=0.25, max_nn=100):
    return o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

source_fpfh = compute_fpfh(source)
target_fpfh = compute_fpfh(target)

# 計算特徵熵
def compute_entropy(fpfh):
    fpfh_data = np.asarray(fpfh.data)
    fpfh_prob = fpfh_data / np.sum(fpfh_data, axis=0, keepdims=True)
    return -np.sum(fpfh_prob * np.log(fpfh_prob + 1e-8), axis=0)

# 找到特徵點
def find_feature_points(point_cloud, fpfh, k=50):
    entropy = compute_entropy(fpfh)  # 使用熵篩選特徵點
    feature_indices = np.argsort(entropy)[:k]  # 選擇熵最低的前 k 個點
    return feature_indices

source_features = find_feature_points(source, source_fpfh, k=50)
target_features = find_feature_points(target, target_fpfh, k=50)

# 標記特徵點
source_colors = np.asarray(source.colors)
target_colors = np.asarray(target.colors)

# 設置整體顏色為白色
source.paint_uniform_color([1.0, 1.0, 1.0])
target.paint_uniform_color([1.0, 1.0, 1.0])

# 標記特徵點
source_colors[source_features] = [1.0, 0.0, 0.0]  # 紅色表示源點雲的特徵點
target_colors[target_features] = [0.0, 0.0, 1.0]  # 藍色表示目標點雲的特徵點

source_colors[source_features] = [1.0, 0.0, 0.0]  # 紅色表示源點雲的特徵點
target_colors[target_features] = [0.0, 0.0, 1.0]  # 藍色表示目標點雲的特徵點

# 自定義顯示函數，設置背景為黑色
def visualize_with_black_background(point_cloud, title):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)  # 設置視窗標題
    vis.add_geometry(point_cloud)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])  # 設置背景為黑色
    vis.run()
    vis.destroy_window()

# 分別顯示源點雲和目標點雲
print("Displaying source point cloud with feature points...")
visualize_with_black_background(source, "Source Point Cloud")

print("Displaying target point cloud with feature points...")
visualize_with_black_background(target, "Target Point Cloud")
