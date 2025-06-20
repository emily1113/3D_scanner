import open3d as o3d      # 只用於讀取與可視化
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ————————————
# 參數設定
# ————————————
file_path = r"C:\Users\ASUS\Desktop\ICP_TIME\00000_00076 (1).ply"
k = 30                # 鄰域大小，可調整以改變邊緣偵測的敏感度
std_multiplier = 1.5  # 閾值以「平均 + std_multiplier × 標準差」計算

# ————————————
# 1. 讀取點雲（含法線）
# ————————————
pcd = o3d.io.read_point_cloud(file_path)
points = np.asarray(pcd.points)   # (N, 3)
normals = np.asarray(pcd.normals) # (N, 3)

# ————————————
# 2. 建立 KD-Tree 找最近鄰
# ————————————
nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(points)
_, indices = nn.kneighbors(points)  # indices.shape = (N, k+1)

# ————————————
# 3. 計算法向量變化率（平均夾角）
# ————————————
normal_variation = np.zeros(points.shape[0])
for i in range(points.shape[0]):
    neigh_idx = indices[i, 1:]          # 排除自己
    neigh_normals = normals[neigh_idx]  # (k, 3)
    cosines = neigh_normals.dot(normals[i])
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)         # 單位：rad
    normal_variation[i] = angles.mean() # 或改為 angles.max()、angles.std()

# ————————————
# 4. 列印統計資訊，並計算閾值
# ————————————
print("normal_variation  min:", normal_variation.min())
print("normal_variation  max:", normal_variation.max())
print("normal_variation  mean:", normal_variation.mean())
print("normal_variation  std:",  normal_variation.std())

thresh = normal_variation.mean() + std_multiplier * normal_variation.std()
print("使用閾值 (mean + {:.1f}×std) =".format(std_multiplier), thresh)

# ————————————
# 5. 分割邊緣點並輸出數量
# ————————————
edge_indices = np.where(normal_variation > thresh)[0]
edge_count = len(edge_indices)
print("偵測到邊緣點數量：", edge_count)

# ————————————
# 6. 可視化（Open3D）：灰色原點雲 + 紅色邊緣點
#    並在視窗標題顯示偵測到的邊緣點數量
# ————————————
edge_pcd = pcd.select_by_index(edge_indices)
# pcd.paint_uniform_color([0.7, 0.7, 0.7])
edge_pcd.paint_uniform_color([1.0, 0.0, 0.0])

o3d.visualization.draw_geometries(
    [ edge_pcd],
    window_name=f"Detected Edge Points: {edge_count}",
)
