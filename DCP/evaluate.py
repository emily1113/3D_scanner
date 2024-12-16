import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 匯入自訂模組
from dataloader import PointCloudDataset
from model import DCP

# 設置資料夾
data_folder = r'C:\Users\ASUS\Desktop\POINT\3D_scanner\DCP\point_clouds'
model_path = r'C:\Users\ASUS\Desktop\POINT\3D_scanner\DCP\dcp_model.pth'

# 確認資料夾和模型文件是否存在
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"資料夾未找到: {data_folder}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到: {model_path}")

# 加載數據集與數據加載器
test_dataset = PointCloudDataset(data_folder, max_points=40000)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 加載模型
model = DCP()
model.load_state_dict(torch.load(model_path))
model.eval()

# 測試損失計算
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = torch.tensor(0.0)

    with torch.no_grad():
        for sources, targets in test_loader:
            batch_size = len(sources)
            batch_loss = torch.tensor(0.0)

            for i in range(batch_size):
                source = sources[i].unsqueeze(0)  # [1, N, 3]
                target = targets[i].unsqueeze(0)  # [1, N, 3]

                if source.shape[1] != target.shape[1]:
                    print(f"跳過不匹配批次: Source: {source.shape}, Target: {target.shape}")
                    continue

                try:
                    # 前向傳播
                    R_pred, T_pred = model(source, target)

                    # 修正矩陣形狀匹配
                    aligned_source = torch.matmul(source, R_pred.transpose(1, 2)) + T_pred.unsqueeze(1)

                    # 計算損失
                    loss = F.mse_loss(aligned_source, target)
                    print(f"批次損失: {loss.item()}")
                    batch_loss += loss

                except RuntimeError as e:
                    print(f"運行錯誤: {e}, Source: {source.shape}, R_pred: {R_pred.shape}, T_pred: {T_pred.shape}")

            total_loss += batch_loss.item()

    print(f'測試損失: {total_loss:.4f}')

# 開始評估
evaluate_model(model, test_loader)
