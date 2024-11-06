import open3d as o3d
import numpy as np

# 使用 Open3D DemoICPPointClouds 資料集
pcd_data = o3d.data.DemoICPPointClouds()
pcd_original = o3d.io.read_point_cloud(pcd_data.paths[0])  # 讀取第一個點雲檔案

# 創建鏡像點雲的副本
pcd_mirrored = pcd_original.translate((0, 0, 0), relative=False)  # 防止修改原始點雲
points = np.asarray(pcd_mirrored.points)
points[:, 1] *= -1  # 沿 Y 軸鏡像
pcd_mirrored.points = o3d.utility.Vector3dVector(points)

# 將原始和鏡像後的點雲進行視覺化顯示
pcd_original.paint_uniform_color([0, 1, 0])  # 原始點雲顯示為紅色
pcd_mirrored.paint_uniform_color([0, 0, 1])  # 鏡像點雲顯示為綠色

# 顯示兩個點雲
o3d.visualization.draw_geometries([pcd_original, pcd_mirrored], 
                                  window_name="Original and Mirrored Point Clouds",
                                  width=800, height=600,
                                  point_show_normal=False)
