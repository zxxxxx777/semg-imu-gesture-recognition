import torch
from torch.utils.data import Dataset


class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].transpose(0, 1)  # (T,C)->(C,T)
        return x, self.y[idx]
