import open3d as o3d
import numpy as np

# 1. 載入原始點雲
pcd = o3d.io.read_point_cloud(r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\furiren_ALL_nor.ply")

# 2. 顯示原始點雲
o3d.visualization.draw_geometries(
    [pcd],
    window_name="原始點雲",
    width=800, height=600,
    point_show_normal=False
)

# 3. 法向量估計（Poisson 重建前需要法向量）
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
)
pcd.normalize_normals()

# 4. Poisson 表面重建
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9, width=0, scale=1.1, linear_fit=False
)

# 5. 移除低密度面片雜訊
density_vals = np.asarray(densities)
density_threshold = np.quantile(density_vals, 0.01)  # 去掉最低 1% 面片
vertices_to_remove = density_vals < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)

# 6. 從網格重採樣為點雲
pcd_complete = mesh.sample_points_poisson_disk(number_of_points=200000)

# 7. 顯示補全後點雲
o3d.visualization.draw_geometries(
    [pcd_complete],
    window_name="補全後點雲",
    width=800, height=600,
    point_show_normal=False
)

# 8. 原 vs 補全 同場比較
#    將原始點雲染成灰色，補全後點雲渲成紅色
pcd.paint_uniform_color([0.7, 0.7, 0.7])
pcd_complete.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw_geometries(
    [pcd, pcd_complete],
    window_name="原始 vs 補全 點雲比較",
    width=800, height=600,
    point_show_normal=False
)

# 9. 儲存補全後點雲
o3d.io.write_point_cloud(
    r"C:\Users\ASUS\Desktop\output_complete.ply",
    pcd_complete
)

print("補全完成，已儲存到 output_complete.ply")
