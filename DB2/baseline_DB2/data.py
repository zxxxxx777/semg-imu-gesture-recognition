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