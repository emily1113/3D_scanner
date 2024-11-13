import open3d as o3d

# 讀取 PLY 點雲
input_file = "C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00000.ply"
point_cloud = o3d.io.read_point_cloud(input_file)

# 轉換並儲存為 ASCII 格式
output_file = "output_ascii.ply"
o3d.io.write_point_cloud(output_file, point_cloud, write_ascii=True)
print(f"點雲已儲存為 ASCII 格式：{output_file}")
