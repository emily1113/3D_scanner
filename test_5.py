import open3d as o3d
import copy

# 讀取點雲檔案
pcd = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/POINT/red/furiren/point_cloud_00001.ply")

# 複製一份原始點雲作為「處理前」
pcd_before = copy.deepcopy(pcd)

# 對原始點雲估算法向量（處理後）
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=20)
)
pcd_after = copy.deepcopy(pcd)

# 為了讓兩個點雲分開顯示，可以對處理後的點雲做一個平移
# 例如：向 x 軸正方向平移一個點雲寬度的距離
bounding_box = pcd_before.get_axis_aligned_bounding_box()
width = bounding_box.get_extent()[0]
pcd_after.translate((width * 1.2, 0, 0))

# 為了區分顯示，給兩個點雲上不同顏色
pcd_before.paint_uniform_color([1, 0, 0])  # 紅色：處理前
pcd_after.paint_uniform_color([0, 1, 0])   # 綠色：處理後

# 使用 draw_geometries 顯示時，對處理後的點雲開啟顯示法向量
o3d.visualization.draw_geometries(
    [pcd_before, pcd_after],
    window_name="Before (Red) and After (Green) with Normals",
    point_show_normal=True
)
