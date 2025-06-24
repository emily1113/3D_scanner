import open3d as o3d
import numpy as np

def compute_point_cloud_size(points):
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    size = max_xyz - min_xyz
    length, width, height = size
    print(f"長: {length:.4f}")
    print(f"寬: {width:.4f}")
    print(f"高: {height:.4f}")
    return min_xyz, max_xyz, length, width, height

def visualize_point_cloud_with_bbox(points, min_xyz, max_xyz):
    # 建立 Open3D 點雲物件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 建立包圍盒
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_xyz, max_xyz)
    aabb.color = (1, 0, 0)  # 紅色

    # 顯示點雲與包圍盒
    o3d.visualization.draw_geometries([pcd, aabb], window_name="點雲與包絡盒")

if __name__ == "__main__":
    # 讀取點雲 (請改成你的檔案路徑)
    FILE_PATH = r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\FRIEREN_stl.ply"
    pcd = o3d.io.read_point_cloud(FILE_PATH)
    points = np.asarray(pcd.points)

    min_xyz, max_xyz, length, width, height = compute_point_cloud_size(points)
    visualize_point_cloud_with_bbox(points, min_xyz, max_xyz)
