import torch
import torch.nn as nn

class DCP(nn.Module):
    def __init__(self):
        super(DCP, self).__init__()

        # 編碼器: 提取點雲特徵
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),    # 輸入 (x, y, z)
            nn.ReLU(),
            nn.Linear(64, 128),  # 中間層
            nn.ReLU(),
            nn.Linear(128, 256)  # 特徵維度: 256
        )

        # 解碼器: 預測旋轉與平移
        self.decoder = nn.Sequential(
            nn.Linear(256 * 2, 512),  # 拼接來源與目標特徵
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9 + 3)  # 旋轉矩陣 (9) + 平移向量 (3)
        )

    def forward(self, source, target):
        # 提取特徵
        src_features = self.encoder(source)  # [1, N, 256]
        tgt_features = self.encoder(target)  # [1, N, 256]

        # 拼接特徵，確保維度匹配
        features = torch.cat((src_features.mean(dim=1), tgt_features.mean(dim=1)), dim=-1)  # [1, 512]

        # 預測變換
        transform = self.decoder(features)

        # 解碼旋轉與平移
        R = transform[:, :9].view(-1, 3, 3)  # 旋轉矩陣
        T = transform[:, 9:].view(-1, 3)    # 平移向量
        print(f"修正後的模型輸出形狀: R_pred: {R.shape}, T_pred: {T.shape}")
        return R, T
