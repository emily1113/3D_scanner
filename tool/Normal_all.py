import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import glob

def compute_normals(points, k=15, view_point=np.array([0, 0, 0], dtype=np.float64), tolerance=1e-8):
    """
    根據點雲資料計算每個點的法向量，並統一取正方向，同時保證數值穩定性。
    
    :param points: (N, 3) numpy 陣列，代表 N 個點的 x, y, z 座標，型態為 np.float64。
    :param k: 用於計算法向量的鄰域點數量。
    :param view_point: 參考視點，所有法向量都將統一指向該點的外側。
    :param tolerance: 當向量模長低於此容忍值時，認為向量為 0，避免除以 0 的情況。
    :return: (N, 3) numpy 陣列，每一列為正規化並統一方向後的法向量。
    """
    n_points = points.shape[0]
    normals = np.zeros_like(points, dtype=np.float64)
    
    # 建立 k 近鄰查詢模型
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        neighbor_pts = points[indices[i]]
        mean = neighbor_pts.mean(axis=0)
        # 計算協方差矩陣
        cov = np.dot((neighbor_pts - mean).T, (neighbor_pts - mean)) / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 選取最小特徵值所對應的特徵向量作為法向量
        normal = eigenvectors[:, 0]
        
        # 正規化法向量，避免因模長過小導致數值不穩定
        norm_val = np.linalg.norm(normal)
        if norm_val > tolerance:
            normal = normal / norm_val
        else:
            normal = np.array([0, 0, 0], dtype=np.float64)
        
        # 統一法向量方向：使其與從視點到該點的向量同向
        if np.dot(points[i] - view_point, normal) < 0:
            normal = -normal
            
        normals[i] = normal
        
    return normals

def process_point_cloud_file(input_file, output_folder, k=15):
    """
    處理單一點雲檔案：讀取檔案、計算法向量、存檔。
    :param input_file: 輸入的 PLY 檔案路徑。
    :param output_folder: 儲存處理後點雲的資料夾。
    :param k: 用於計算法向量的鄰域點數量。
    """
    # 讀取點雲並轉成 double 型態
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points).astype(np.float64)
    
    # 設定參考視點（此處以原點為例）
    view_point = np.array([0, 0, 0], dtype=np.float64)
    
    # 計算法向量
    normals = compute_normals(points, k=k, view_point=view_point)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 將點雲顏色設為黑色
    pcd.paint_uniform_color([0, 0, 0])
    
    # 設定輸出檔名（例如在原檔名前加上 "normals_"）
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_folder, "normals_" + base_name)
    
    # 存檔（包含法向量資訊）
    o3d.io.write_point_cloud(output_file, pcd)
    print("已處理檔案：", input_file)
    print("結果已儲存至：", output_file)

def process_folder(input_folder, output_folder, k=15):
    """
    處理指定資料夾內所有的 PLY 檔案。
    :param input_folder: 存放點雲 PLY 檔案的資料夾。
    :param output_folder: 儲存處理後檔案的資料夾。
    :param k: 用於計算法向量的鄰域點數量。
    """
    # 如果輸出資料夾不存在，則建立它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 搜尋所有 PLY 檔案
    file_pattern = os.path.join(input_folder, "*.ply")
    file_list = glob.glob(file_pattern)
    
    # 逐一處理每個檔案
    for file in file_list:
        process_point_cloud_file(file, output_folder, k=k)

# 設定輸入資料夾與輸出資料夾路徑
input_folder = "C:/Users/ASUS/Desktop/POINT/red/ICP_5_cut/"
output_folder = "C:/Users/ASUS/Desktop/POINT/red/ICP_5_cut/processed/"

# 執行資料夾內所有檔案的處理
process_folder(input_folder, output_folder, k=15)
