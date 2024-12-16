import torch
from torch.utils.data import Dataset
import os

class PointCloudDataset(Dataset):
    def __init__(self, folder, max_points=40000):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pt")]
        self.max_points = max_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        source = torch.load(self.files[idx])
        target = self.random_transform(source)

        # 確保維度匹配
        source = self.pad_points(source)
        target = self.pad_points(target)
        return source, target

    def random_transform(self, point_cloud):
        R = torch.eye(3) + 0.1 * torch.randn(3, 3)  # 隨機旋轉
        T = 0.1 * torch.randn(1, 3)  # 隨機平移
        return point_cloud @ R.T + T

    def pad_points(self, point_cloud):
        if len(point_cloud) < self.max_points:
            padding = self.max_points - len(point_cloud)
            pad_points = torch.zeros(padding, 3)
            point_cloud = torch.cat([point_cloud, pad_points], dim=0)
        else:
            point_cloud = point_cloud[:self.max_points]
        return point_cloud
