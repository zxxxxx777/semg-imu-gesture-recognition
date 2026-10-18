import torch
import torch.nn as nn
from config import DROPOUT_CNN, DROPOUT_HEAD, DROPOUT_LSTM


class AttnPool(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.w = nn.Linear(feat_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.w(h)
        a = torch.softmax(a, dim=1)
        return (a * h).sum(dim=1)


class EMGHandNet1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout_cnn: float = DROPOUT_CNN,
        dropout_head: float = DROPOUT_HEAD,
        dropout_lstm: float = DROPOUT_LSTM,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_cnn),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
        )
        self.pool = AttnPool(feat_dim=2 * lstm_hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_head),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        z = self.cnn(x)
        z = z.transpose(1, 2)
        h, _ = self.lstm(z)
        features = self.pool(h)
        logits = self.head(features)
        return features, logits
