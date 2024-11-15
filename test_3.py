import open3d as o3d

# 讀取點雲檔案
pcd0 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1_40/point_cloud_00000.ply")
pcd1 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1_40/point_cloud_00001.ply")
pcd2 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1_40/point_cloud_00002.ply")
pcd3 = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/1_40/point_cloud_00003.ply")

# 設定顏色
pcd0.paint_uniform_color([1, 0, 0])  # 紅色
pcd1.paint_uniform_color([0, 1, 0])  # 綠色
pcd2.paint_uniform_color([0, 0, 1])  # 藍色
pcd3.paint_uniform_color([1, 1, 0])  # 黃色

# 計算每個點雲的中心點
center0 = pcd0.get_center()
center1 = pcd1.get_center()
center2 = pcd2.get_center()
center3 = pcd3.get_center()

# 將每個點雲平移，使中心點位於原點
pcd0.translate(-center0)
pcd1.translate(-center1)
pcd2.translate(-center2)
pcd3.translate(-center3)

# 合併點雲
merged_pcd = pcd0 + pcd1 + pcd2 + pcd3

# 保存對齊後的點雲到檔案
output_path = "C:/Users/ASUS/Desktop/POINT/aligned_point_cloud.ply"
o3d.io.write_point_cloud(output_path, merged_pcd)
print(f"已將對齊後的點雲保存到: {output_path}")

