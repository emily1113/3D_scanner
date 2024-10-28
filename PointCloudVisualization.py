import open3d as o3d

# # 讀取點雲文件（例如 .ply, .pcd 格式）
# pcd = o3d.io.read_point_cloud("path_to_your_point_cloud_file.ply")

# # 顯示點雲
# o3d.visualization.draw_geometries([pcd])



knot_mesh = o3d.data.ArmadilloMesh().path
mesh = o3d.io.read_triangle_mesh(knot_mesh)
o3d.visualization.draw_geometries([mesh])