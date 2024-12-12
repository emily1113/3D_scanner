import numpy as np
import open3d as o3d
from sklearn.neighbors import BallTree

def compute_harris_response_with_balltree(pcd, knn=30, k=0.04, min_response=1e-6, num_keypoints=100):
    """
    計算 Harris3D 響應，使用 BallTree 加速最近鄰搜索

    Args:
        pcd: 點雲
        knn: 每個點的鄰域大小
        k: Harris 響應公式中的參數
        min_response: 最小 Harris 響應閾值
        num_keypoints: 要提取的特徵點數量

    Returns:
        list: 包含 (響應值, 點座標) 的列表，按響應值降序排序
    """

    # 設定法向量估算
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(k=knn)

    # 將點雲轉換為 NumPy 陣列
    points = np.asarray(pcd.points)

    # 建立 BallTree
    ball_tree = BallTree(points)

    # 初始化 Harris 響應列表
    harris_response = []

    for i, point in enumerate(points):
        # 使用 BallTree 查找最近鄰
        _, idx = ball_tree.query([point], k=knn)
        idx = idx[0]  # 取出最近鄰索引

        # 計算結構張量
        cov_matrix = np.zeros((3, 3))
        for j in idx:
            diff = points[j] - point
            cov_matrix += np.outer(diff, diff)

        # 計算特徵值
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        lambda1, lambda2, lambda3 = sorted(eigenvalues)

        # 計算 Harris 響應
        response = lambda3 - k * (lambda1 + lambda2) ** 2
        if response >= min_response:
            harris_response.append((response, point))

    # 根據響應值排序
    return sorted(harris_response, key=lambda x: x[0], reverse=True)[:num_keypoints]


# 設定參數
knn = 50
k = 0.05
min_response = 1e-6  # 降低最小響應閾值
num_keypoints = 500
ply_file_path = "C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply"

# 載入點雲數據
pcd = o3d.io.read_point_cloud(ply_file_path)

# 將點雲設為白色
pcd.paint_uniform_color([1, 1, 1])

# 計算 Harris3D 響應
harris_points = compute_harris_response_with_balltree(pcd, knn=knn, k=k, min_response=min_response, num_keypoints=num_keypoints)

# 顯示結果
keypoints = [point for _, point in harris_points]
keypoints_pcd = o3d.geometry.PointCloud()
keypoints_pcd.points = o3d.utility.Vector3dVector(np.array(keypoints))
keypoints_pcd.paint_uniform_color([1, 0, 0])  # 紅色特徵點

# 視覺化
o3d.visualization.draw_geometries([pcd, keypoints_pcd])

# 釋放記憶體與清理暫存資源
del pcd
del keypoints_pcd
import gc
gc.collect()
