import numpy as np
from scipy.spatial import Delaunay
from plyfile import PlyData, PlyElement

def read_ply_xyz(filename):
    """用 plyfile 讀取 PLY 點雲，回傳 (N,3) numpy 陣列"""
    ply = PlyData.read(filename)
    vertex = ply['vertex']
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    return pts

def write_ply_mesh(filename, vertices, triangles):
    """將三角網格寫成 PLY 檔"""
    # 準備頂點結構
    vert_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    verts = np.array([tuple(v) for v in vertices], dtype=vert_dtype)
    # 準備三角面結構 (每個面含 3 個頂點 index)
    face_dtype = [('vertex_indices', 'i4', (3,))]
    faces = np.array([([a,b,c],) for a,b,c in triangles], dtype=face_dtype)

    el_verts = PlyElement.describe(verts, 'vertex')
    el_faces = PlyElement.describe(faces, 'face')
    PlyData([el_verts, el_faces], text=True).write(filename)

def alpha_shape_3d(points: np.ndarray, alpha: float):
    """
    計算 3D 點雲的 alpha shape 外部邊界三角面。

    Args:
        points (np.ndarray): shape (N,3) 的點座標
        alpha (float): 外接球半徑上限，越大包裹越鬆

    Returns:
        vertices (np.ndarray): 外殼頂點 (M,3)
        triangles (List[Tuple[int,int,int]]): 三角面 index
    """
    # 1. Delaunay 四面體剖分
    delaunay = Delaunay(points)
    tets = delaunay.simplices  # (n_tet,4)

    # 2. 計算每個 tetrahedron 的外接球半徑
    def circumsphere_radius(pa, pb, pc, pd):
        A = np.vstack([pb-pa, pc-pa, pd-pa]).T  # 3×3
        b = np.array([
            np.dot(pb-pa, pb-pa),
            np.dot(pc-pa, pc-pa),
            np.dot(pd-pa, pd-pa)
        ])
        try:
            x = np.linalg.solve(A, b)
            center = pa + 0.5 * x
            return np.linalg.norm(center - pa)
        except np.linalg.LinAlgError:
            return np.inf

    radii = np.array([
        circumsphere_radius(*points[tet])
        for tet in tets
    ])

    # 3. 篩選小於 alpha 的 tetrahedron
    good = tets[radii < alpha]

    # 4. 收集所有三角面並只保留出現一次的（外殼）
    face_count = {}
    for tet in good:
        # 4 faces per tet
        faces = [
            tuple(sorted((tet[i], tet[j], tet[k])))
            for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
        ]
        for f in faces:
            face_count[f] = face_count.get(f, 0) + 1

    boundary = [f for f,c in face_count.items() if c == 1]

    # 5. 重建頂點索引
    vids = sorted({v for f in boundary for v in f})
    vid_map = {old:i for i,old in enumerate(vids)}
    vertices = points[vids]
    triangles = [(vid_map[a], vid_map[b], vid_map[c]) for a,b,c in boundary]

    return vertices, triangles

if __name__ == "__main__":
    # 1. 讀取原始點雲
    pts = read_ply_xyz(r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\furiren_ALL.ply")

    # 2. 設定 alpha（可依據雲圖尺度試調）
    diameter = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    alpha = 0.08 * diameter

    # 3. 計算外殼
    verts, tris = alpha_shape_3d(pts, alpha)

    # 4. 寫出外殼 Mesh
    write_ply_mesh(r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\alpha_shell1.ply", verts, tris)

    print(f"輸出 {len(verts)} 個頂點，{len(tris)} 個三角面 到 alpha_shell.ply")
