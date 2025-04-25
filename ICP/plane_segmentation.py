import open3d as o3d
import numpy as np

def find_horizontal_plane(pcd, axis=np.array([0,0,1]), angle_thresh_deg=10,
                          distance_threshold=0.005, ransac_n=3, num_iterations=1000,
                          max_planes=5):
    """
    反覆執行 plane segmentation，挑法向量與 axis 夾角小於 threshold 的平面。
    回傳 (plane_model, inlier_indices)；找不到回傳 (None, None)。
    """
    down = pcd.voxel_down_sample(voxel_size=0.01)
    remaining = down
    axis = axis / np.linalg.norm(axis)

    for i in range(max_planes):
        plane, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        normal = np.array(plane[:3])
        normal /= np.linalg.norm(normal)
        angle = np.degrees(np.arccos(np.abs(normal.dot(axis))))
        print(f"第 {i+1} 平面，法向量={normal}, 與軸夾角={angle:.1f}°")

        if angle < angle_thresh_deg:
            return plane, inliers

        # 移除剛才找到的那一塊，繼續下一輪
        remaining = remaining.select_by_index(inliers, invert=True)

    return None, None

# ===== 主流程 =====
pcd = o3d.io.read_point_cloud(r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\furiren_ALL_nor.ply")
plane_model, inliers = find_horizontal_plane(pcd)
if plane_model is None:
    print("沒找到合適的水平平面")
else:
    a,b,c,d = plane_model
    print(f"選到水平平面：{a:.4f}x+{b:.4f}y+{c:.4f}z+{d:.4f}=0")
    # 分離並可視化
    ground = pcd.select_by_index(inliers)
    rest   = pcd.select_by_index(inliers, invert=True)
    ground.paint_uniform_color([1,0,0])
    rest.paint_uniform_color([0.7,0.7,0.7])
    o3d.visualization.draw_geometries([rest, ground], window_name="挑選出的底平面")
