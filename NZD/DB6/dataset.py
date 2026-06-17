import torch
from torch.utils.data import Dataset


class ACCEMGWindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 兼容旧 train.py / 旧 notebook 的名字。
FMGEMGWindowDataset = ACCEMGWindowDataset
