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

def visualize_with_bbox_and_axes_and_corners(points, min_xyz, max_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    aabb = o3d.geometry.AxisAlignedBoundingBox(min_xyz, max_xyz)
    aabb.color = (1, 0, 0)

    # XYZ 座標軸
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    # 原點小球
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    origin_sphere.paint_uniform_color([0, 0, 0])
    origin_sphere.translate([0, 0, 0])

    # 包絡盒八個頂點
    bbox_points = np.asarray(aabb.get_box_points())
    spheres = []
    for i, pt in enumerate(bbox_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        sphere.paint_uniform_color([0, 0, 1])  # 藍色
        sphere.translate(pt)
        spheres.append(sphere)

    o3d.visualization.draw_geometries([pcd, aabb, axis, origin_sphere] + spheres, window_name="點雲+包絡盒+座標軸+頂點")
    # print 座標
    print("包絡盒八個頂點座標：")
    for i, pt in enumerate(bbox_points):
        print(f"頂點 {i+1}: {pt}")

if __name__ == "__main__":
    FILE_PATH = r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\furiren_ALL.ply"
    pcd = o3d.io.read_point_cloud(FILE_PATH)
    points = np.asarray(pcd.points)

    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    visualize_with_bbox_and_axes_and_corners(points, min_xyz, max_xyz)