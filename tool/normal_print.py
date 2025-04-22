import open3d as o3d


def display_point_cloud_normals(pcd_path, radius=0.05, max_nn=30):
    """
    讀取點雲並以線段方式顯示其法向量。

    參數:
        pcd_path: str, 點雲檔案路徑
        radius: float, 法向量估計時的搜尋半徑
        max_nn: int, 法向量估計時的最大鄰域點數
    """
    # 1. 讀取點雲
    pcd = o3d.io.read_point_cloud(pcd_path)

    # 2. 估計法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn))

    # 3. 以線段方式呈現法向量
    o3d.visualization.draw_geometries(
        [pcd],
        window_name='Point Cloud Normals',
        point_show_normal=True  # 顯示法向量線段
    )


if __name__ == "__main__":
    FILE_PATH = r"C:\Users\ASUS\Desktop\POINT\red\furiren\result\furiren_ALL_nor.ply"
    display_point_cloud_normals(
        FILE_PATH,
        radius=0.1,
        max_nn=30
    )
