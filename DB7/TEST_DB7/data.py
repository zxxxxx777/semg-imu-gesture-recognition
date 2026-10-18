import torch
from torch.utils.data import Dataset


class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) #将输入数据转换为PyTorch的张量格式，方便后续训练和评估
        self.y = torch.tensor(y, dtype=torch.long)   #标签数据转换为长整型，适用于分类任务的损失函数要求
    def __len__(self):
        return len(self.X)   #DataLoader会用它来决定epoch 里有多少个样本

    def __getitem__(self, idx):
        x = self.X[idx].transpose(0, 1)  # (T,C)->(C,T)
        return x, self.y[idx]

class MultiModalDataset(Dataset):
    def __init__(self, X_emg, X_imu, y):
        self.X_emg = torch.tensor(X_emg, dtype=torch.float32)
        self.X_imu = torch.tensor(X_imu, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  #y是标签

    def __len__(self):
        return len(self.X_emg)

    def __getitem__(self, idx):
        x_emg = self.X_emg[idx].transpose(0, 1)#transpose进行了时间维度的交换，将时间维度作为第一个维度
        x_imu = self.X_imu[idx].transpose(0, 1)
        return x_emg, x_imu, self.y[idx]