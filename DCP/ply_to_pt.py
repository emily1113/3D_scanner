import torch
import open3d as o3d
import os

# 修正成完整路徑
ply_folder = r'C:\Users\ASUS\Desktop\POINT\3D_scanner\DCP\point_clouds'

# 確認資料夾是否存在
if not os.path.exists(ply_folder):
    os.makedirs(ply_folder)
    print(f"資料夾已創建: {ply_folder}")

# 將 PLY 檔案轉換為 PT 檔案
def ply_to_tensor(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    points = torch.tensor(pcd.points, dtype=torch.float32)
    return points

# 轉換點雲文件
for file in os.listdir(ply_folder):
    if file.endswith(".ply"):
        points = ply_to_tensor(os.path.join(ply_folder, file))
        torch.save(points, os.path.join(ply_folder, file.replace(".ply", ".pt")))
        print(f"成功轉換: {file}")
