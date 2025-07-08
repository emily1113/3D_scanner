import numpy as np
import open3d as o3d
import time

# ---------------------------------------------
# 參數設定
# ---------------------------------------------
INPUT_PATH = r"C:\Users\user\Desktop\PointCloud\red\test\normals_point_cloud_00000.ply"
VOXEL_SIZE = 0.5  # 單位同點雲座標
VISUALIZE = True
# ---------------------------------------------

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    自訂體素下採樣函式。
    參數:
      points: (N,3) numpy 陣列, 原始點雲座標
      voxel_size: float, 體素邊長, 單位與點雲單位相同
    回傳:
      (M,3) numpy 陣列, 下採樣後點雲
    """
    min_bounds = np.min(points, axis=0)
    voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(np.int32)

    voxel_dict = {}
    for pt, idx in zip(points, voxel_indices):
        key = tuple(idx)
        voxel_dict.setdefault(key, []).append(pt)

    down_pts = []
    for pts in voxel_dict.values():
        pts_arr = np.asarray(pts)
        down_pts.append(pts_arr.mean(axis=0))

    return np.asarray(down_pts)


def main():
    # 計時：讀取點雲
    t_read_start = time.perf_counter()
    pcd = o3d.io.read_point_cloud(INPUT_PATH)
    points = np.asarray(pcd.points)
    t_read_end = time.perf_counter()

    # 計時：體素下採樣
    t_down_start = time.perf_counter()
    down_pts = voxel_downsample(points, VOXEL_SIZE)
    t_down_end = time.perf_counter()

    # 建立並計時：寫出點雲
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(down_pts)
    t_write_start = time.perf_counter()
    # 如果你要寫檔，可在這裡呼叫 write
    # o3d.io.write_point_cloud(output_path, down_pcd)
    t_write_end = time.perf_counter()

    # 結果輸出
    print(f"---------- 執行結果 ----------")
    print(f"原始點數: {len(points)}")
    print(f"下採樣後點數: {len(down_pts)}")
    print()
    print(f"讀取耗時   : {(t_read_end - t_read_start):.4f} 秒")
    print(f"下採樣耗時 : {(t_down_end - t_down_start):.4f} 秒")
    print(f"寫出耗時   : {(t_write_end - t_write_start):.4f} 秒")
    print(f"總耗時     : {(t_write_end - t_read_start):.4f} 秒")
    print(f"體素格子大小: {VOXEL_SIZE}")

    # 顯示結果（用紅色）
    if VISUALIZE:
        down_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 全局紅色
        o3d.visualization.draw_geometries(
            [down_pcd],
            window_name="Voxel Downsampled (Red)",
            width=800,
            height=600
        )

if __name__ == "__main__":
    main()
