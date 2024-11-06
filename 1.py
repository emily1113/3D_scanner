import open3d as o3d
import numpy as np

# 載入 Demo ICP 點雲數據集
pcd_data = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(pcd_data.paths[0])
target = o3d.io.read_point_cloud(pcd_data.paths[1])

# 為了更好地區分兩個點雲，為他們設置不同顏色
source.paint_uniform_color([1, 0, 0])  # source 點雲設為紅色
target.paint_uniform_color([0, 1, 0])  # target 點雲設為綠色

# 定義 90 度的旋轉矩陣（沿著 Y 軸）
R = source.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))  # 90 度轉換為弧度
source.rotate(R, center=(0, 0, 0))  # 圍繞原點進行旋轉

# 添加坐標系
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

# 可視化旋轉後的 source 和 target 點雲以及坐標系
o3d.visualization.draw_geometries([source, target, coordinate_frame],
                                  window_name="沿 Y 軸旋轉後的 Source 和 Target 點雲 可視化",
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50,
                                  point_show_normal=False)
