import numpy as np
import open3d as o3d
import time

# ---------------------------------------------
# 參數設定
# ---------------------------------------------
INPUT_PATH    = r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00000.ply"
AVERAGE_K     = 5      # 每隔 AVERAGE_K 個點取一個點
VISUALIZE     = True
# ---------------------------------------------

def average_downsample(points: np.ndarray, k: int) -> np.ndarray:
    """
    平均下採樣：每隔 k 個點取一個點。
    參數:
      points: (N,3) numpy 陣列, 原始點雲座標
      k: int, 取樣間隔
    回傳:
      (M,3) numpy 陣列, 下採樣後點雲
    """
    return points[::k, :]

def main():
    # 讀取點雲
    t_read_start = time.perf_counter()
    pcd    = o3d.io.read_point_cloud(INPUT_PATH)
    points = np.asarray(pcd.points)
    t_read_end = time.perf_counter()

    # 平均下採樣
    t_down_start = time.perf_counter()
    down_pts = average_downsample(points, AVERAGE_K)
    t_down_end  = time.perf_counter()

    # 建立下採樣後的 PointCloud 物件
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(down_pts)

    # (可選) 保留原本顏色，若要統一紅色請用 paint_uniform_color
    down_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 全部標成紅色

    # 輸出結果
    print("----- 平均下採樣結果 -----")
    print(f"原始點數:          {len(points)}")
    print(f"下採樣後點數:      {len(down_pts)}  (every {AVERAGE_K} points)")
    print(f"讀取耗時:          {(t_read_end - t_read_start):.4f} 秒")
    print(f"平均下採樣耗時:    {(t_down_end - t_down_start):.4f} 秒")
    print(f"取樣間隔 (k) =     {AVERAGE_K}")

    # 顯示
    if VISUALIZE:
        o3d.visualization.draw_geometries(
            [down_pcd],
            window_name=f"平均下採樣 (每 {AVERAGE_K} 點)",
            width=800,
            height=600
        )

if __name__ == "__main__":
    main()
