import open3d as o3d
import numpy as np
import os

# 設定點雲資料夾與輸出資料夾
source_folder = "C:/Users/ASUS/Desktop/POINT/red/furiren/"
output_folder = "C:/Users/ASUS/Desktop/POINT/red/furiren/output/"

# 建立輸出資料夾
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ICP 配準函式
def run_icp(source, target, voxel_size=0.05, threshold=0.01, max_iteration=2000):
    # 下採樣點雲
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # 計算法向量
    radius_normal = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))

    # 執行點對平面 ICP
    trans_init = np.identity(4)
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )

    return icp_result.transformation

# 批次進行多點雲配準
for i in range(76):  # 00000 ~ 00075
    source_file = os.path.join(source_folder, f"point_cloud_{i:05d}.ply")
    target_file = os.path.join(source_folder, f"point_cloud_{i+1:05d}.ply")
    output_file = os.path.join(output_folder, f"aligned_point_cloud_{i:05d}.ply")

    if os.path.exists(source_file) and os.path.exists(target_file):
        source = o3d.io.read_point_cloud(source_file)
        target = o3d.io.read_point_cloud(target_file)

        print(f"正在處理：{source_file} 對 {target_file}")

        transformation = run_icp(source, target)
        print(f"ICP 變換矩陣 ({i}):")
        print(transformation)

        # 應用變換並儲存配準結果
        source.transform(transformation)
        o3d.io.write_point_cloud(output_file, source)
        print(f"完成配準並儲存：{output_file}")
    else:
        print(f"檔案不存在：{source_file} 或 {target_file}")

# 顯示配準後的點雲
aligned_files = sorted(os.listdir(output_folder))
aligned_clouds = [o3d.io.read_point_cloud(os.path.join(output_folder, f)) for f in aligned_files]

print("顯示所有配準後的點雲")
o3d.visualization.draw_geometries(aligned_clouds, window_name="ICP 多點雲配準結果")
