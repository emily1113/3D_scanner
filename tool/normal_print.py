import open3d as o3d
import numpy as np
import time

def display_point_cloud_normals_colored(pcd_path, normal_length=0.1, radius=0.1, max_nn=30):
    """
    讀取點雲並以紅色線段方式顯示其法向量，同時顯示各步驟的計算耗時。

    參數:
        pcd_path: str, 點雲檔案路徑
        normal_length: float, 法向量線段的長度
        radius: float, 法向量估計時的搜尋半徑
        max_nn: int, 法向量估計時的最大鄰域點數
    """
    # 開始總計時
    t0 = time.perf_counter()

    # 1. 讀取點雲
    t1 = time.perf_counter()
    pcd = o3d.io.read_point_cloud(pcd_path)
    t2 = time.perf_counter()
    print(f"讀取點雲耗時：{(t2 - t1):.4f} 秒")

    # 2. 估計法向量
    t3 = time.perf_counter()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    t4 = time.perf_counter()
    print(f"估計法向量耗時：{(t4 - t3):.4f} 秒")

    # 3. 構造線段 (LineSet) 表示法向量
    t5 = time.perf_counter()
    pts = np.asarray(pcd.points)
    norms = np.asarray(pcd.normals)

    line_pts = []
    lines = []
    for i, (p, n) in enumerate(zip(pts, norms)):
        start = p
        end = p + n * normal_length
        line_pts.extend([start, end])
        lines.append([2*i, 2*i+1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_pts))
    line_set.lines  = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in lines])
    t6 = time.perf_counter()
    print(f"構造 LineSet 耗時：{(t6 - t5):.4f} 秒")

    # 顯示總耗時
    t7 = time.perf_counter()
    print(f"整體流程總耗時：{(t7 - t0):.4f} 秒")

    # 4. 同時顯示點雲與紅色法向量線段
    o3d.visualization.draw_geometries(
        [pcd, line_set],
        window_name='Point Cloud Normals (Red)'
    )

if __name__ == "__main__":
    FILE_PATH = r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00000.ply"
    display_point_cloud_normals_colored(
        FILE_PATH,
        normal_length=5,  # 法向量線段長度
        radius=0.1,       # 法向量搜尋半徑
        max_nn=30         # 最大鄰域點數
    )
