import os
import open3d as o3d
import numpy as np
import matplotlib.cm as cm

# 設定資料夾路徑為你指定的路徑
folder_path = r'C:\Users\ASUS\Desktop\POINT\red\furiren\processed\0~77'

# 篩選資料夾中的點雲檔案，假設檔案格式為 .pcd 或 .ply
file_list = [f for f in os.listdir(folder_path) if f.endswith('.pcd') or f.endswith('.ply')]
file_list.sort()  # 假設排序後就是從淺到深

num_files = len(file_list)
if num_files == 0:
    raise ValueError("資料夾中沒有找到符合的點雲檔案。")

# 初始化一個空的點雲，用於存放合併後的結果
merged_cloud = o3d.geometry.PointCloud()

# 逐一讀取點雲、賦予顏色、合併並顯示
for index, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    
    # 讀取目前點雲
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 計算色彩映射參數 (0～1)
    t = index / (num_files - 1) if num_files > 1 else 0
    # 取得 jet colormap 之 RGB 值（只取前 3 個元素）
    rgb = cm.jet(t)[:3]
    
    # 為目前點雲建立與其點數相同的顏色陣列
    points = np.asarray(pcd.points)
    colors = np.tile(rgb, (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 合併目前點雲到總結果中
    merged_cloud += pcd
    
    # 顯示目前合併的結果，每次顯示會阻斷，需手動關閉視窗才能繼續
    print(f"目前合併進度：{index + 1}/{num_files} - {file_name}")
    o3d.visualization.draw_geometries([merged_cloud])

# 合併完成後，將最終結果存檔成 ply 檔案
o3d.io.write_point_cloud("merged_point_cloud.ply", merged_cloud)
print("合併結果已存檔為 'merged_point_cloud.ply'")
