import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from plyfile import PlyData

def read_ply_file(file_path):
    """
    使用 plyfile 讀取 PLY 檔案，回傳點座標與法向量（若存在）。
    """
    plydata = PlyData.read(file_path)
    vertex_data = plydata['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    if {'nx', 'ny', 'nz'}.issubset(vertex_data.dtype.names):
        normals = np.vstack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']]).T
    else:
        normals = None  # 若沒有法向量，需自行計算（此處不做處理）
    return points, normals

def compute_spfh(points, normals, search_radius, nbins=11):
    """
    計算每個點的 SPFH 特徵，透過統計鄰居間的三個角度特徵。
    輸出為 (N, 3*nbins) 的直方圖向量，其中 N 為點數。
    
    特徵計算流程（對於點 p 與鄰居 q）：
      1. 計算 d = (q - p) 並正規化得到 d_unit
      2. 以 p 處法向量 n_p 作為 u，計算 v = normalize(cross(u, d_unit)) 與 w = cross(u, v)
      3. 計算三個特徵：
         - f1 = v · n_q        (範圍約為 [-1,1])
         - f2 = u · d_unit     (範圍約為 [-1,1])
         - f3 = arctan2(w · n_q, u · n_q)   (範圍 [-π, π])
    將每個特徵依各自範圍劃分為 nbins 個 bin，最後將三個直方圖串接並正規化。
    """
    N = points.shape[0]
    spfh = np.zeros((N, 3*nbins))
    tree = cKDTree(points)
    
    # 設定各特徵的直方圖 bin 邊界
    bins_f1 = np.linspace(-1, 1, nbins+1)
    bins_f2 = np.linspace(-1, 1, nbins+1)
    bins_f3 = np.linspace(-np.pi, np.pi, nbins+1)
    
    for i in range(N):
        p = points[i]
        n_p = normals[i]
        # 找出半徑內鄰居（不包含自己）
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
            
            # 建立 Darboux 座標系
            u = n_p  # 假設 n_p 已歸一化
            v = np.cross(u, d_unit)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-6:
                continue
            v = v / norm_v
            w = np.cross(u, v)
            
            # 計算三個特徵
            f1 = np.dot(v, n_q)         # [-1,1]
            f2 = np.dot(u, d_unit)      # [-1,1]
            f3 = np.arctan2(np.dot(w, n_q), np.dot(u, n_q))  # [-pi, pi]
            
            # 累加直方圖
            bin_idx = np.searchsorted(bins_f1, f1, side='right') - 1
            if 0 <= bin_idx < nbins:
                hist_f1[bin_idx] += 1
            bin_idx = np.searchsorted(bins_f2, f2, side='right') - 1
            if 0 <= bin_idx < nbins:
                hist_f2[bin_idx] += 1
            bin_idx = np.searchsorted(bins_f3, f3, side='right') - 1
            if 0 <= bin_idx < nbins:
                hist_f3[bin_idx] += 1
        
        # 將三個直方圖串接後 L1 正規化
        hist = np.concatenate([hist_f1, hist_f2, hist_f3])
        total = np.sum(hist)
        if total > 0:
            spfh[i] = hist / total
        else:
            spfh[i] = hist
    return spfh

def compute_fpfh(points, normals, search_radius, nbins=11):
    """
    根據 SPFH 特徵與鄰居資訊，計算 FPFH 特徵：
      FPFH(p) = SPFH(p) + (1/k) * Σ[ (1/||p - q||) * SPFH(q) ]
    """
    spfh = compute_spfh(points, normals, search_radius, nbins)
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

#------------------- 使用範例 -------------------
if __name__ == '__main__':
    file_path = "C:/Users/ASUS/Desktop/POINT/red/FPFH/5/point_cloud_with_normals_cut_0.ply"
    points, normals = read_ply_file(file_path)
    if normals is None:
        print("點雲中沒有法向量資訊，請先計算法向量！")
    else:
        # 設定鄰域搜尋半徑（根據點雲尺度調整）
        search_radius = 0.2
        fpfh_features = compute_fpfh(points, normals, search_radius, nbins=11)
        print("計算得到的 FPFH 特徵維度：", fpfh_features.shape)
        # 顯示第一個點的 FPFH 特徵
        print("第一個點的 FPFH 特徵：\n", fpfh_features[0])

        # 簡單視覺化直方圖（例如：第一個點）
        plt.figure(figsize=(8,4))
        plt.bar(np.arange(fpfh_features.shape[1]), fpfh_features[0])
        plt.xlabel("Histogram Bin")
        plt.ylabel("Normalized Frequency")
        plt.title("FPFH for the first point")
        plt.show()
