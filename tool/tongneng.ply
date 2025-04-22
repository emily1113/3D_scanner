from plyfile import PlyData
import open3d as o3d

def load_bunny_data():
    # 使用 Open3D 內建的 Bunny 模型數據（Stanford Bunny）
    bunny_source = o3d.data.BunnyMesh().path  # 獲取模型文件的路徑
    print("Bunny 模型文件位置:", bunny_source)
    
    # 讀取三角網格模型
    mesh = o3d.io.read_triangle_mesh(bunny_source)
    
    # 計算每個頂點的法線
    mesh.compute_vertex_normals()
    
    # 從網格中使用 Poisson Disk 採樣生成點雲（這裡採樣 1000 個點）
    point_cloud = mesh.sample_points_poisson_disk(1000)
    return point_cloud

# 加載 Bunny 點雲數據
point_cloud = load_bunny_data()

# 將點雲寫入 PLY 檔案，再使用 PlyData 讀取
o3d.io.write_point_cloud("bunny_point_cloud.ply", point_cloud)
ply1 = PlyData.read("bunny_point_cloud.ply")
ply2 = PlyData.read("C:/Users/ASUS/Desktop/POINT/red/furiren/double.ply")

# 比較 header
print(ply1.header)
print(ply2.header)

# 比較頂點數據
vertices1 = ply1['vertex'].data
vertices2 = ply2['vertex'].data
print("檔案1頂點數：", len(vertices1))
print("檔案2頂點數：", len(vertices2))
