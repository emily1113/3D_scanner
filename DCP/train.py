import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import PointCloudDataset
from model import DCP

# 設定資料夾
data_folder = r'C:\Users\ASUS\Desktop\POINT\3D_scanner\DCP\point_clouds'
train_dataset = PointCloudDataset(data_folder, max_points=40000)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 初始化模型與優化器
model = DCP()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 訓練函數
def train_model(model, train_loader, optimizer, epochs=150):
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for sources, targets in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, requires_grad=True)

            for i in range(len(sources)):
                source = sources[i].unsqueeze(0)  # [1, N, 3]
                target = targets[i].unsqueeze(0)  # [1, N, 3]

                # 檢查點雲形狀是否匹配
                if source.shape[1] != target.shape[1]:
                    print(f"跳過批次，來源點雲形狀: {source.shape}, 目標點雲形狀: {target.shape}")
                    continue

                try:
                    # 前向傳播
                    R_pred, T_pred = model(source, target)

                    # 修正矩陣形狀匹配
                    aligned_source = torch.matmul(source, R_pred.transpose(1, 2)) + T_pred.unsqueeze(1)

                    # 計算損失
                    loss = F.mse_loss(aligned_source, target)
                    batch_loss += loss

                except RuntimeError as e:
                    print(f"運行錯誤: {e}, 來源形狀: {source.shape}, R_pred: {R_pred.shape}, T_pred: {T_pred.shape}")

            # 反向傳播與優化
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

# 開始訓練
train_model(model, train_loader, optimizer)
