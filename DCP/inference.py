import os
import torch
import open3d as o3d
import numpy as np
from model import DCP

# 設置模型和點雲文件路徑
model_path = r'C:\Users\ASUS\Desktop\POINT\3D_scanner\DCP\dcp_model.pth'
source_ply = r'C:\Users\ASUS\Desktop\POINT\3D_scanner\DCP\point_clouds\furiren_0.ply'
target_ply = r'C:\Users\ASUS\Desktop\POINT\3D_scanner\DCP\point_clouds\furiren_19.ply'
max_points = 40000  # 固定最大點數

# 加載模型
model = DCP()
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到: {model_path}")
model.load_state_dict(torch.load(model_path))
model.eval()
print("模型已成功加載！")

# 加載點雲文件 (修剪或填充)
def load_point_cloud(file_path, max_points):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"點雲文件未找到: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    print(f"加載 {file_path} 點雲形狀: {points.shape}")

    # 修剪或填充點雲
    if points.shape[0] > max_points:
        points = points[:max_points, :]
    else:
        padding = max_points - points.shape[0]
        points = np.vstack([points, np.zeros((padding, 3))])
    return torch.tensor(points, dtype=torch.float32)

# 可視化點雲
def visualize(source, target, aligned):
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    aligned_pcd = o3d.geometry.PointCloud()

    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd.points = o3d.utility.Vector3dVector(target)
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned)

    source_pcd.paint_uniform_color([1, 0, 0])   # 紅色: 原始來源點雲
    target_pcd.paint_uniform_color([0, 1, 0])   # 綠色: 目標點雲
    aligned_pcd.paint_uniform_color([0, 0, 1])  # 藍色: 配準結果

    o3d.visualization.draw_geometries([source_pcd, target_pcd, aligned_pcd])

# 加載來源與目標點雲
source = load_point_cloud(source_ply, max_points).unsqueeze(0)  # [1, N, 3]
target = load_point_cloud(target_ply, max_points).unsqueeze(0)  # [1, N, 3]
print(f"修剪後的來源點雲形狀: {source.shape}, 目標點雲形狀: {target.shape}")

# 推論
with torch.no_grad():
    R_pred, T_pred = model(source, target)
    print(f"修正後的模型輸出形狀: R_pred: {R_pred.shape}, T_pred: {T_pred.shape}")

    # 檢查矩陣形狀是否匹配
    if R_pred.shape != torch.Size([1, 3, 3]):
        raise RuntimeError(f"點雲形狀不匹配，無法進行推論: {R_pred.shape} vs {source.shape}")

    aligned_source = torch.matmul(source, R_pred.transpose(1, 2)) + T_pred.unsqueeze(1)
    print(f"配準結果形狀: {aligned_source.shape}")

# 視覺化結果
visualize(source.squeeze(0).numpy(), target.squeeze(0).numpy(), aligned_source.squeeze(0).numpy())
