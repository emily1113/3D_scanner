import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])


def point_cloud_registration(source, target, threshold=1.0, init_transformation=np.eye(4)):
    # 使用點對平面的ICP算法進行點雲配準
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transformation, p2l)
    return reg_p2l.transformation


def merge_point_clouds(point_clouds, transformations):
    merged_pcd = o3d.geometry.PointCloud()
    for i, pcd in enumerate(point_clouds):
        temp_pcd = copy.deepcopy(pcd)
        temp_pcd.transform(transformations[i])
        merged_pcd += temp_pcd
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)
    return merged_pcd


if __name__ == "__main__":
    # 讀取不同角度的點雲
    pcd_data = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(pcd_data.paths[0])
    target = o3d.io.read_point_cloud(pcd_data.paths[1])
    
    # 初始轉換矩陣（可以根據情況調整）
    init_transformation = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                      [-0.139, 0.967, -0.215, 0.7],
                                      [0.487, 0.255, 0.835, -1.4],
                                      [0.0, 0.0, 0.0, 1.0]])

    # 進行點雲配準
    print("Performing ICP registration...")
    transformation = point_cloud_registration(source, target, threshold=1.0, init_transformation=init_transformation)
    print("Transformation matrix:\n", transformation)

    # 顯示配準結果
    draw_registration_result(source, target, transformation)

    # 合併點雲
    merged_pcd = merge_point_clouds([source, target], [np.eye(4), transformation])

    # 保存最終的合併點雲
    o3d.io.write_point_cloud("merged_point_cloud.ply", merged_pcd)
    print("最終點雲已保存到 'merged_point_cloud.ply'")
