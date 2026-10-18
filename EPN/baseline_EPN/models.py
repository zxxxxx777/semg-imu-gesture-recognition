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


class FeatureFusionNet(nn.Module):
    def __init__(
        self,
        emg_ch: int,
        imu_ch: int,
        num_classes: int,
        lstm_hidden: int = 128,
        dropout_head: float = DROPOUT_HEAD,
    ):
        super().__init__()
        self.emg_encoder = EMGHandNet1D(
            in_ch=emg_ch,
            num_classes=num_classes,
            lstm_hidden=lstm_hidden,
        )
        self.imu_encoder = EMGHandNet1D(
            in_ch=imu_ch,
            num_classes=num_classes,
            lstm_hidden=lstm_hidden,
        )

        concat_dim = (2 * lstm_hidden) * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_head),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_emg, x_imu):
        feat_emg, logit_emg = self.emg_encoder(x_emg)
        feat_imu, logit_imu = self.imu_encoder(x_imu)
        fused_feat = torch.cat([feat_emg, feat_imu], dim=1)
        fusion_logits = self.fusion_mlp(fused_feat)
        return fusion_logits, logit_emg, logit_imu
