import numpy as np
import open3d as o3d
import copy  # 用於深度拷貝
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from plyfile import PlyData
import random
import time  # 為了延時顯示

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
    print(f"降採樣後點雲點數量: {down_sampled_point_count}")
    
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
      5. 對每個特徵根據指定 bin 數 (nbins) 進行直方圖累計，最後串接並正規化直方圖
    
    輸出:
      spfh : (N, 3*nbins) 的 SPFH 特徵向量
    """
    N = points.shape[0]
    spfh = np.zeros((N, 3*nbins))
    tree = cKDTree(points)
    
    # 設定三個特徵的直方圖 bin 邊界
    bins_f1 = np.linspace(-1, 1, nbins+1)
    bins_f2 = np.linspace(-1, 1, nbins+1)
    bins_f3 = np.linspace(-np.pi, np.pi, nbins+1)
    
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
            u = n_p  # u 為 p 處法向量
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

def numpy_to_o3d_feature(np_features):
    """
    將 numpy 陣列轉換為 Open3D Feature 物件
    Open3D 的 Feature 物件要求 data 的 shape 為 (feature_dim, num_points)
    """
    feature = o3d.pipelines.registration.Feature()
    feature.data = np_features.T.astype(np.float64)
    return feature

# ------------------ 自訂 SAC-IA 配準（不使用 Open3D 內建函式） ------------------

def estimate_rigid_transform(A, B):
    """
    根據來源點集合 A 與目標點集合 B（均為 n x 3 陣列）計算剛性變換 T (4x4)
    使用 SVD 分解法求解:
      1. 計算 A 與 B 的質心
      2. 將點去質心後構成中心化矩陣
      3. 計算協方差矩陣，並進行 SVD 分解
      4. 根據 U, Vt 計算旋轉矩陣 R，若行列式為負則調整
      5. 計算平移向量 t = centroid_B - R * centroid_A
    回傳 4x4 的變換矩陣 T
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T.dot(BB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # 保證 R 為正旋轉矩陣（determinant = 1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T.dot(U.T)
    t = centroid_B - R.dot(centroid_A)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def sac_ia_registration_custom(source_points, target_points, source_features, target_features,
                               max_iterations=1000, inlier_threshold=0.05, ransac_n=4):
    """
    自訂 SAC-IA 配準，不使用 Open3D 的內建函式。
    
    流程：
      1. 利用 KDTree 在目標特徵空間中為每個來源特徵找出最近鄰 (建立特徵對應)
      2. 利用 RANSAC 隨機採樣 ransac_n 組對應點，根據 SVD 估計剛性變換
      3. 對所有來源點依據該變換計算內點數 (inlier: 變換後的來源點與目標點距離小於 inlier_threshold)
      4. 重複多次後選出內點數最多的變換作為最終結果
      
    輸出:
      result : 字典，包含 'transformation' (4x4 變換矩陣) 與 'inlier_count'
    """
    # 先在目標特徵中建立 KDTree
    tree = cKDTree(target_features)
    # 為每個來源特徵找最近鄰
    distances, nn_indices = tree.query(source_features, k=1)
    # 建立對應列表：每個來源點對應到目標點
    correspondences = [(i, nn_indices[i]) for i in range(len(source_points))]
    
    best_inlier_count = 0
    best_transformation = np.eye(4)
    
    for i in range(max_iterations):
        # 隨機抽取 ransac_n 個對應點
        sample = random.sample(correspondences, ransac_n)
        src_idx = [s for s, t in sample]
        tgt_idx = [t for s, t in sample]
        A = source_points[src_idx]  # 來源點集合
        B = target_points[tgt_idx]    # 目標點集合
        # 估計剛性變換
        T = estimate_rigid_transform(A, B)
        # 將所有來源點依據 T 轉換
        src_points_homo = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
        transformed_src = (T.dot(src_points_homo.T)).T[:, :3]
        
        inlier_count = 0
        # 對每個對應點，計算變換後來源點與目標點的歐氏距離，若小於門檻則視為內點
        for s_idx, t_idx in correspondences:
            p_trans = transformed_src[s_idx]
            q = target_points[t_idx]
            if np.linalg.norm(p_trans - q) < inlier_threshold:
                inlier_count += 1
        
        # 若此變換的內點數較多，則更新最佳變換
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_transformation = T
            # 可選：若內點數達到某比例，可提前中斷
            if best_inlier_count > 0.8 * len(correspondences):
                break

    result = {"transformation": best_transformation, "inlier_count": best_inlier_count}
    return result

# ------------------ ICP 精細配準（此部分先註解掉） ------------------
# def refine_registration(source, target, initial_transformation, distance_threshold):
#     """
#     使用 ICP (Iterative Closest Point) 進行精細配準，這裡採用點到平面方法。
#     此部分仍使用 Open3D 的內建函式進行 ICP 配準。
#     """
#     result_icp = o3d.pipelines.registration.registration_icp(
#         source, target,
#         max_correspondence_distance=distance_threshold,
#         init=initial_transformation,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
#     )
#     return result_icp

def display_point_clouds(*point_clouds, title="Point Clouds"):
    """
    使用 Open3D 視覺化點雲，參數 point_clouds 為點雲列表
    """
    o3d.visualization.draw_geometries(list(point_clouds), window_name=title)

# ------------------ 主程序 ------------------

if __name__ == "__main__":
    # 參數設定
    sample_step = 2          # 每隔 2 個點取一個進行下採樣
    voxel_size = 0.01         # 用於配準門檻設定（不直接用於 FPFH 計算）
    search_radius = 0.05      # 自訂 FPFH 計算中的鄰域搜尋半徑
    distance_threshold = voxel_size * 3.0  # SAC-IA 配準門檻（此處用於 RANSAC 中內點判定）
    max_iterations = 4000    # RANSAC 最大迭代次數
    inlier_threshold = 0.03  # 內點距離門檻
    ransac_n = 4             # 每次 RANSAC 隨機採樣的對應點數

    # 指定 source 與 target 點雲檔案路徑
    source_file = "C:/Users/ASUS/Desktop/POINT/red/FPFH/5/point_cloud_with_normals_cut_0.ply"
    target_file = "C:/Users/ASUS/Desktop/POINT/red/FPFH/5/point_cloud_with_normals_cut_3.ply"

    # 讀取點雲
    source = load_point_cloud(source_file)
    target = load_point_cloud(target_file)

    # 若原始點雲缺少法向量則估計法向量
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 均勻下採樣
    source_down = preprocess_point_cloud(source, sample_step)
    target_down = preprocess_point_cloud(target, sample_step)

    # 若下採樣後缺少法向量則重新估計
    if not source_down.has_normals():
        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if not target_down.has_normals():
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # 取得下採樣後點雲的點座標與法向量（轉換成 numpy 陣列）
    points_source = np.asarray(source_down.points)
    normals_source = np.asarray(source_down.normals)
    points_target = np.asarray(target_down.points)
    normals_target = np.asarray(target_down.normals)

    # 設定直方圖的 bin 數
    nbins = 11
    # 使用自訂 FPFH 計算
    source_fpfh_np = compute_fpfh_custom(points_source, normals_source, search_radius, nbins)
    target_fpfh_np = compute_fpfh_custom(points_target, normals_target, search_radius, nbins)

    # 將 numpy 計算得到的 FPFH 特徵轉換為 Open3D Feature 物件（若後續需要使用 Open3D ICP）
    source_fpfh = numpy_to_o3d_feature(source_fpfh_np)
    target_fpfh = numpy_to_o3d_feature(target_fpfh_np)

    # ------------------ 標示特徵點 ------------------
    # 以每個點的 FPFH 能量 (L2 範數) 作為評估依據，選取前 10% 能量最高的點作為特徵點

    ## Source 部分
    fpfh_norms_source = np.linalg.norm(source_fpfh_np, axis=1)
    num_keypoints_source = int(0.1 * len(fpfh_norms_source))
    if num_keypoints_source < 1:
        num_keypoints_source = 1
    keypoint_indices_source = np.argsort(fpfh_norms_source)[-num_keypoints_source:]
    print("Source 選取的特徵點索引:", keypoint_indices_source)

    keypoints_source_pcd = o3d.geometry.PointCloud()
    keypoints_source_pcd.points = o3d.utility.Vector3dVector(points_source[keypoint_indices_source])
    keypoints_source_pcd.paint_uniform_color([0, 0, 1])
    # 將下採樣點雲標記為灰色
    source_down.paint_uniform_color([0.8, 0.8, 0.8])

    ## Target 部分
    fpfh_norms_target = np.linalg.norm(target_fpfh_np, axis=1)
    num_keypoints_target = int(0.1 * len(fpfh_norms_target))
    if num_keypoints_target < 1:
        num_keypoints_target = 1
    keypoint_indices_target = np.argsort(fpfh_norms_target)[-num_keypoints_target:]
    print("Target 選取的特徵點索引:", keypoint_indices_target)

    keypoints_target_pcd = o3d.geometry.PointCloud()
    keypoints_target_pcd.points = o3d.utility.Vector3dVector(points_target[keypoint_indices_target])
    keypoints_target_pcd.paint_uniform_color([0, 0, 1])
    target_down.paint_uniform_color([0.8, 0.8, 0.8])

    # 暫停 3 秒後依序顯示 source 與 target 的特徵點標示
    time.sleep(5)
    o3d.visualization.draw_geometries([source_down, keypoints_source_pcd],
                                      window_name="Source FPFH 特徵點標示",
                                      width=1600, height=1200)
    time.sleep(5)
    o3d.visualization.draw_geometries([target_down, keypoints_target_pcd],
                                      window_name="Target FPFH 特徵點標示",
                                      width=1600, height=1200)

    # ------------------ 配準與視覺化 ------------------
    # 配準前視覺化，將 source 著紅色，target 著綠色
    source_down.paint_uniform_color([1, 0, 0])
    target_down.paint_uniform_color([0, 1, 0])
    display_point_clouds(source_down, target_down, title="Before Registration (Custom FPFH)")

    # 使用自訂 SAC-IA 進行初始對齊（完全不依賴 Open3D 內建配準函式）
    sac_result = sac_ia_registration_custom(points_source, points_target, source_fpfh_np, target_fpfh_np,
                                             max_iterations=max_iterations,
                                             inlier_threshold=inlier_threshold,
                                             ransac_n=ransac_n)
    print("自訂 SAC-IA 初始對齊結果:")
    print("內點數:", sac_result["inlier_count"])
    print("對齊變換矩陣:")
    print(sac_result["transformation"])

    # 將自訂 SAC-IA 得到的變換應用於 source 下採樣點雲
    source_down.transform(sac_result["transformation"])
    # 配準後著色為藍色
    source_down.paint_uniform_color([0, 0, 1])
    display_point_clouds(source_down, target_down, title="After Registration (Custom SAC-IA)")

    # ICP 精細配準部分先註解掉
    # refined_distance_threshold = voxel_size * 1.5
    # result_icp = refine_registration(source_down, target_down, sac_result["transformation"], refined_distance_threshold)
    # print("ICP 精細配準結果:")
    # print(result_icp)
    # print("ICP 對齊變換矩陣:")
    # print(result_icp.transformation)
    #
    # # 將 ICP 精細配準結果應用於 source 點雲並視覺化
    # source_down.transform(result_icp.transformation)
    # source_down.paint_uniform_color([0, 0, 1])
    # display_point_clouds(source_down, target_down, title="After Registration (Refined ICP)")
