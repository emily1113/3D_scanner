import open3d as o3d
import numpy as np
import math
import copy  # 用於深拷貝點雲

# -------- 使用者參數設定 --------
INPUT_PATH = r"C:\Users\user\Desktop\PointCloud\red\furiren_ALL.ply"
# 變換後輸出檔案路徑
OUTPUT_PATH = r"C:\Users\user\Desktop\PointCloud\red\furiren_ALL_1.ply"
ROTATION_DEG = [30.0, 45.0, 60.0]
TRANSLATION = [100.0, 0.0, 50.0]
# --------------------------------

def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    print(f"讀入點雲，點數: {len(pcd.points)}")
    return pcd

def transform_point_cloud(pcd, rotation_deg, translation):
    rotation_rad = [math.radians(a) for a in rotation_deg]
    R = o3d.geometry.get_rotation_matrix_from_xyz(rotation_rad)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = translation
    # 深拷貝一份點雲再做變換
    pcd2 = copy.deepcopy(pcd)
    pcd2.transform(T)
    print(f"已對點雲應用旋轉 {rotation_deg} 度 & 平移 {translation}")
    return pcd2

def visualize(pcd_list, window_name="Point Cloud"):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([*pcd_list, axis], window_name=window_name)

def main():
    pcd_orig = load_point_cloud(INPUT_PATH)
    pcd_tf   = transform_point_cloud(pcd_orig, ROTATION_DEG, TRANSLATION)
    visualize([pcd_orig, pcd_tf], window_name="原始(灰) vs 變換後(彩)")
    # 以下為寫檔功能，如需開啟請移除下面兩行的註解
    # o3d.io.write_point_cloud(OUTPUT_PATH, pcd_tf)
    # print(f"變換後點雲已儲存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
