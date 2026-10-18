import torch
from torch.utils.data import Dataset


class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].transpose(0, 1)
        return x, self.y[idx]


class MultiModalDataset(Dataset):
    def __init__(self, X_emg, X_imu, y):
        self.X_emg = torch.tensor(X_emg, dtype=torch.float32)
        self.X_imu = torch.tensor(X_imu, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X_emg)

    def __getitem__(self, idx):
        x_emg = self.X_emg[idx].transpose(0, 1)
        x_imu = self.X_imu[idx].transpose(0, 1)
        return x_emg, x_imu, self.y[idx]
