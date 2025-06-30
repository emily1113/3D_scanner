import open3d as o3d
import numpy as np

corner_colors = [
    [1, 0, 0],    # 0: 紅
    [0, 1, 0],    # 1: 綠
    [0, 0, 1],    # 2: 藍
    [1, 1, 0],    # 3: 黃
    [1, 0, 1],    # 4: 紫
    [0, 1, 1],    # 5: 青
    [1, 0.5, 0],  # 6: 橘
    [0.5, 0.5, 0.5]  # 7: 灰
]
color_names = ['紅', '綠', '藍', '黃', '紫', '青', '橘', '灰']

def visualize_with_bbox_and_colored_corners_and_bottom_face(points, min_xyz, max_xyz):
    # 計算長寬高
    size = max_xyz - min_xyz
    length, width, height = size

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    aabb = o3d.geometry.AxisAlignedBoundingBox(min_xyz, max_xyz)
    aabb.color = (1, 0, 0)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    origin_sphere.paint_uniform_color([0, 0, 0])
    origin_sphere.translate([0, 0, 0])

    bbox_points = np.asarray(aabb.get_box_points())
    spheres = []
    for i, pt in enumerate(bbox_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.paint_uniform_color(corner_colors[i])
        sphere.translate(pt)
        spheres.append(sphere)

    # 底面（紫4、青5、藍2、灰7）
    bottom_face_vertices = np.array([bbox_points[4], bbox_points[5], bbox_points[2], bbox_points[7]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh_bottom = o3d.geometry.TriangleMesh()
    mesh_bottom.vertices = o3d.utility.Vector3dVector(bottom_face_vertices)
    mesh_bottom.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_bottom.paint_uniform_color([0.6, 0.2, 0.8])
    mesh_bottom.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [pcd, aabb, axis, origin_sphere, mesh_bottom] + spheres,
        window_name="包絡盒底面填色（紫青藍灰）"
    )

    # 終端機輸出
    print("包絡盒八個頂點座標與顏色：")
    for i, pt in enumerate(bbox_points):
        print(f"頂點 {i+1}（{color_names[i]}）: {pt}")
    print("\n包絡盒長寬高：")
    print(f"長: {length:.4f}")
    print(f"寬: {width:.4f}")
    print(f"高: {height:.4f}")

if __name__ == "__main__":
    # FILE_PATH = r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\furiren_ALL.ply"
    FILE_PATH = r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\FRIEREN_stl_nor.ply"
    pcd = o3d.io.read_point_cloud(FILE_PATH)
    points = np.asarray(pcd.points)

    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    visualize_with_bbox_and_colored_corners_and_bottom_face(points, min_xyz, max_xyz)
