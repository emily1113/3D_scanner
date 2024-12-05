import open3d as o3d

# 讀取點雲
ply_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/point_cloud_00001.ply"
point_cloud = o3d.io.read_point_cloud(ply_path)

# 使用 Open3D 的可視化工具來手動選擇感興趣的點
print("請在視窗中選取感興趣的點，按下 'P' 來選擇點並按 'Q' 關閉視窗")
o3d.visualization.draw_geometries_with_editing([point_cloud])

# 保存選取的點雲
output_path = "C:/Users/ASUS/Desktop/POINT/red/ICP_5/selected_points.ply"
o3d.io.write_point_cloud(output_path, point_cloud)
