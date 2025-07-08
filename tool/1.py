import open3d as o3d
import time

# 讀取點雲（含 RGB 色彩）
FILE_PATH = r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00000.ply"
pcd = o3d.io.read_point_cloud(FILE_PATH)

# 如果你想忽略原本色彩，改用灰色顯示：
pcd.paint_uniform_color([0.5, 0.5, 0.5])  # RGB 全域灰

o3d.visualization.draw_geometries(
    [pcd],
    window_name="原始點雲（灰色）",
    width=800,
    height=600,
    left=50,
    top=50,
    point_show_normal=False
)

# 設定體素尺寸
voxel_size = 1.0

# 計時並下採樣
start = time.perf_counter()
down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
end = time.perf_counter()

print(f"原始點數：{len(pcd.points)}")
print(f"下採樣後點數：{len(down_pcd.points)}")
print(f"下採樣耗時：{(end - start):.4f} 秒")

# 把下採樣後的點雲塗成紅色
down_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 紅色

o3d.visualization.draw_geometries(
    [down_pcd],
    window_name=f"下採樣後點雲（紅色, voxel_size={voxel_size}）",
    width=800,
    height=600,
    left=900,
    top=50,
    point_show_normal=False
)
