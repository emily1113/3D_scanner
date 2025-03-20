import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from plyfile import PlyData
import time

# ------------------ 讀取與下採樣 ------------------

def load_point_cloud(file_path):
    """
    使用 Open3D 讀取點雲檔案，回傳點雲物件
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def preprocess_point_cloud(point_cloud, sample_step):
    """
    自訂均勻下採樣：利用 numpy 陣列切片 (每隔 sample_step 個點取一個)
    同時保留法向量與顏色資訊

    參數:
      point_cloud : 原始點雲（Open3D 格式）
      sample_step : 下採樣步長，越小保留的點越多

    回傳:
      downsampled_pcd : 下採樣後的點雲（Open3D 格式）
    """
    # 取得原始點雲點數量
    original_point_count = len(point_cloud.points)
    print(f"原始點雲點數量: {original_point_count}")

    # 將點雲轉換成 numpy 陣列後，使用陣列切片進行均勻下採樣
    points = np.asarray(point_cloud.points)
    sampled_points = points[::sample_step]

    # 建立新的點雲物件並設定取樣後的點
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

    # 若原始點雲有法向量，則同樣進行下採樣
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        sampled_normals = normals[::sample_step]
        downsampled_pcd.normals = o3d.utility.Vector3dVector(sampled_normals)

    # 若原始點雲有顏色資訊，則同樣進行下採樣
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        sampled_colors = colors[::sample_step]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

    down_sampled_point_count = len(downsampled_pcd.points)
    print(f"下採樣後點雲點數量: {down_sampled_point_count}")
    return downsampled_pcd

# ------------------ 自訂 FPFH 計算（不使用 Open3D 內建函式） ------------------

def compute_spfh_custom(points, normals, search_radius, nbins=11):
    """
    計算每個點的 SPFH 特徵（簡化版）

    流程：
      1. 利用 cKDTree 找出每個點在 search_radius 內的鄰居 (不包含自身)
      2. 對於每一對 (p, q)，計算向量 d = (q - p) 與其單位向量 d_unit
      3. 以 p 處的法向量 n_p 作為 u，計算 v = normalize(cross(u, d_unit)) 與 w = cross(u, v)
      4. 計算三個特徵：
           f1 = dot(v, n_q)      (範圍約為 [-1,1])
           f2 = dot(u, d_unit)   (範圍約為 [-1,1])
           f3 = arctan2(dot(w, n_q), dot(u, n_q))  (範圍為 [-π, π])
      5. 將三個特徵各自依 nbins 個 bin 累計直方圖後串接，並進行 L1 正規化

    輸出:
      spfh : (N, 3*nbins) 的 SPFH 特徵向量
    """
    N = points.shape[0]
    spfh = np.zeros((N, 3 * nbins))
    tree = cKDTree(points)

    # 設定三個特徵的直方圖 bin 邊界
    bins_f1 = np.linspace(-1, 1, nbins + 1)
    bins_f2 = np.linspace(-1, 1, nbins + 1)
    bins_f3 = np.linspace(-np.pi, np.pi, nbins + 1)

    for i in range(N):
        p = points[i]
        n_p = normals[i]
        # 找出在指定半徑內的鄰居（排除自己）
        idx = tree.query_ball_point(p, search_radius)
        idx = [j for j in idx if j != i]
        if len(idx) == 0:
            continue
        hist_f1 = np.zeros(nbins)
        hist_f2 = np.zeros(nbins)
        hist_f3 = np.zeros(nbins)
        for j in idx:
            q = points[j]
            n_q = normals[j]
            d = q - p
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-6:
                continue
            d_unit = d / d_norm

            # 建立局部 Darboux 座標系
            u = n_p  # u 為 p 處的法向量
            v = np.cross(u, d_unit)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-6:
                continue
            v = v / norm_v
            w = np.cross(u, v)

            # 計算三個特徵值
            f1 = np.dot(v, n_q)
            f2 = np.dot(u, d_unit)
            f3 = np.arctan2(np.dot(w, n_q), np.dot(u, n_q))

            # 將 f1 累加到直方圖中
            bin_idx = np.searchsorted(bins_f1, f1, side='right') - 1
            if 0 <= bin_idx < nbins:
                hist_f1[bin_idx] += 1
            # 將 f2 累加到直方圖中
            bin_idx = np.searchsorted(bins_f2, f2, side='right') - 1
            if 0 <= bin_idx < nbins:
                hist_f2[bin_idx] += 1
            # 將 f3 累加到直方圖中
            bin_idx = np.searchsorted(bins_f3, f3, side='right') - 1
            if 0 <= bin_idx < nbins:
                hist_f3[bin_idx] += 1

        # 串接三個直方圖，並做 L1 正規化
        hist = np.concatenate([hist_f1, hist_f2, hist_f3])
        total = np.sum(hist)
        if total > 0:
            spfh[i] = hist / total
        else:
            spfh[i] = hist
    return spfh

def compute_fpfh_custom(points, normals, search_radius, nbins=11):
    """
    根據 SPFH 特徵與鄰居資訊，計算 FPFH 特徵
    公式：
      FPFH(p) = SPFH(p) + (1/k) * Σ[ (1/||p - q||) * SPFH(q) ]
    """
    spfh = compute_spfh_custom(points, normals, search_radius, nbins)
    N = points.shape[0]
    fpfh = np.copy(spfh)
    tree = cKDTree(points)
    for i in range(N):
        p = points[i]
        idx = tree.query_ball_point(p, search_radius)
        idx = [j for j in idx if j != i]
        if len(idx) == 0:
            continue
        weighted_sum = np.zeros_like(spfh[i])
        weight_total = 0.0
        for j in idx:
            d = np.linalg.norm(points[j] - p)
            if d < 1e-6:
                continue
            weight = 1.0 / d
            weighted_sum += weight * spfh[j]
            weight_total += weight
        if weight_total > 0:
            fpfh[i] += weighted_sum / weight_total
    return fpfh

# ------------------ 主程式 ------------------

if __name__ == "__main__":
    # 參數設定
    sample_step = 5         # 每隔 5 個點取一個
    search_radius = 0.1     # FPFH 計算時使用的鄰域搜尋半徑
    nbins = 11              # 直方圖的 bin 數

    # ------------------ 使用 Bunny 模型 ------------------
    # 利用 Open3D 內建資料集載入 Stanford Bunny 網格模型，
    # 並均勻取樣轉換成點雲（此處取樣 2000 個點，可依需求調整）
    bunny_data = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny_data.path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=2000)
    print("使用 Bunny 模型，點雲點數:", len(pcd.points))

    # 若點雲缺少法向量，則計算法向量
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius * 2, max_nn=30))

    # 均勻下採樣
    pcd_down = preprocess_point_cloud(pcd, sample_step)

    # 取得下採樣後點雲的點與法向量（轉換成 numpy 陣列）
    points = np.asarray(pcd_down.points)
    normals = np.asarray(pcd_down.normals)

    # 使用自訂 FPFH 計算
    fpfh_features = compute_fpfh_custom(points, normals, search_radius, nbins)

    # 輸出 FPFH 特徵的形狀（形狀為 (點數, 3*nbins)）
    print("FPFH 特徵形狀:", fpfh_features.shape)

    # 輸出第一個點的 FPFH 特徵向量
    print("第一個點的 FPFH 特徵:")
    print(fpfh_features[0])

    # 使用 matplotlib 繪製第一個點的 FPFH 直方圖
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(fpfh_features.shape[1]), fpfh_features[0])
    plt.xlabel("Histogram Bin")
    plt.ylabel("Normalized Frequency")
    plt.title("FPFH 直方圖 (第一個點)")
    plt.show()

    # ------------------ 標示特徵點 ------------------
    # 這裡以每個點的 FPFH 能量 (L2 範數) 作為評估依據，
    # 選取前 10% 能量最高的點作為特徵點
    fpfh_norms = np.linalg.norm(fpfh_features, axis=1)
    num_keypoints = int(0.1 * len(fpfh_norms))
    if num_keypoints < 1:
        num_keypoints = 1
    keypoint_indices = np.argsort(fpfh_norms)[-num_keypoints:]
    print("選取的特徵點索引:", keypoint_indices)

    # 創建特徵點點雲 (標示為藍色)
    keypoints_pcd = o3d.geometry.PointCloud()
    keypoints_pcd.points = o3d.utility.Vector3dVector(points[keypoint_indices])
    keypoints_pcd.paint_uniform_color([1, 0, 0])

    # 下採樣點雲顯示為灰色
    pcd_down.paint_uniform_color([0.8, 0.8, 0.8])
    
    # 暫停數秒後進行視覺化
    time.sleep(3)
    o3d.visualization.draw_geometries([keypoints_pcd, pcd_down],
                                      window_name="FPFH 特徵點標示",
                                      width=1600, height=1200)
