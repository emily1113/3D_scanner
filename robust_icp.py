# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
# """Outlier rejection using robust kernels for ICP"""

import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # 黃色
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 藍色
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00000.ply")
    target = o3d.io.read_point_cloud("C:/Users/ASUS/Desktop/ICP/ICP/red/1_40/point_cloud_00001.ply")
    
    # 計算法向量
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    trans_init = np.array([[0, -1, 0, 0],
                           [1,  0, 0, 0],
                           [0,  0, 1, 0],
                           [0,  0, 0, 1]])

    # 加入噪聲
    mu, sigma = 0, 0.1
    source_noisy = apply_noise(source, mu, sigma)

    print("Displaying source point cloud + noise:")
    o3d.visualization.draw([source_noisy])

    print("Displaying original source and target point cloud with initial transformation:")
    draw_registration_result(source, target, trans_init)

    threshold = 1.0
    print("Using the noisy source pointcloud to perform robust ICP.\n")
    print("Robust point-to-plane ICP, threshold={}:".format(threshold))
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    print("Using robust loss:", loss)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_noisy, target, threshold, trans_init, p2l)
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)

# 合併點雲
combined_pcd = source + target

# 將轉換應用到原始的來源點雲後加入合併的點雲
transformed_source = copy.deepcopy(source)
transformed_source.transform(reg_p2l.transformation)
combined_pcd = transformed_source + target

# 保存合併的點雲
o3d.io.write_point_cloud("combined_point_cloud.ply", combined_pcd)
print("合併的點雲已保存到 'combined_point_cloud.ply'")