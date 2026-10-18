import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    ALIGN_PROJ_DIM,
    CROSS_ATTN_DROPOUT,
    CROSS_ATTN_HEADS,
    DROPOUT_CNN,
    DROPOUT_HEAD,
    DROPOUT_LSTM,
    MAX_CROSS_RES_SCALE,
    MODALITY_DROPOUT_P,
)


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
        in_ch: int = 12,
        num_classes: int = 40,
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

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        z = self.cnn(x)
        z = z.transpose(1, 2)
        h, _ = self.lstm(z)
        features = self.pool(h)
        logits = self.head(features)
        if return_sequence:
            return features, logits, h
        return features, logits


class BiResidualCrossAttention(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_residual_scale: float = 0.2,
    ):
        super().__init__()
        self.max_residual_scale = max_residual_scale
        self.emg_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.imu_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.emg_norm = nn.LayerNorm(feat_dim)
        self.imu_norm = nn.LayerNorm(feat_dim)
        self.emg_pool = AttnPool(feat_dim)
        self.imu_pool = AttnPool(feat_dim)
        self.emg_delta = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feat_dim, feat_dim),
        )
        self.imu_delta = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(feat_dim, feat_dim),
        )
        self.scale_emg = nn.Linear(feat_dim * 4, feat_dim)
        self.scale_imu = nn.Linear(feat_dim * 4, feat_dim)

        for module in (self.emg_delta[-1], self.imu_delta[-1], self.scale_emg, self.scale_imu):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        seq_emg: torch.Tensor,
        seq_imu: torch.Tensor,
        feat_emg: torch.Tensor,
        feat_imu: torch.Tensor,
    ):
        emg_from_imu_seq, _ = self.emg_attn(query=seq_emg, key=seq_imu, value=seq_imu, need_weights=False)
        imu_from_emg_seq, _ = self.imu_attn(query=seq_imu, key=seq_emg, value=seq_emg, need_weights=False)

        emg_cross_seq = self.emg_norm(seq_emg + emg_from_imu_seq)
        imu_cross_seq = self.imu_norm(seq_imu + imu_from_emg_seq)
        emg_cross = self.emg_pool(emg_cross_seq)
        imu_cross = self.imu_pool(imu_cross_seq)

        delta_emg = self.emg_delta(torch.cat([feat_emg, emg_cross], dim=1))
        delta_imu = self.imu_delta(torch.cat([feat_imu, imu_cross], dim=1))

        context = torch.cat(
            [feat_emg, feat_imu, torch.abs(feat_emg - feat_imu), feat_emg * feat_imu],
            dim=1,
        )
        gate_emg = self.max_residual_scale * torch.sigmoid(self.scale_emg(context))
        gate_imu = self.max_residual_scale * torch.sigmoid(self.scale_imu(context))
        enhanced_emg = feat_emg + gate_emg * delta_emg
        enhanced_imu = feat_imu + gate_imu * delta_imu
        return enhanced_emg, enhanced_imu, gate_emg, gate_imu


class FeatureFusionNet(nn.Module):
    def __init__(
        self,
        emg_ch: int,
        imu_ch: int,
        num_classes: int = 40,
        lstm_hidden: int = 128,
        dropout_head: float = DROPOUT_HEAD,
    ):
        super().__init__()
        self.emg_encoder = EMGHandNet1D(in_ch=emg_ch, num_classes=num_classes, lstm_hidden=lstm_hidden)
        self.imu_encoder = EMGHandNet1D(in_ch=imu_ch, num_classes=num_classes, lstm_hidden=lstm_hidden)

        feat_dim = 2 * lstm_hidden
        concat_dim = feat_dim * 4
        self.cross_residual = BiResidualCrossAttention(
            feat_dim=feat_dim,
            num_heads=CROSS_ATTN_HEADS,
            dropout=CROSS_ATTN_DROPOUT,
            max_residual_scale=MAX_CROSS_RES_SCALE,
        )
        self.modality_dropout_p = MODALITY_DROPOUT_P
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_head),
            nn.Linear(256, num_classes),
        )
        self.emg_align_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, ALIGN_PROJ_DIM),
        )
        self.imu_align_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, ALIGN_PROJ_DIM),
        )

    def forward(self, x_emg, x_imu):
        feat_emg, logit_emg, seq_emg = self.emg_encoder(x_emg, return_sequence=True)
        feat_imu, logit_imu, seq_imu = self.imu_encoder(x_imu, return_sequence=True)
        enhanced_emg, enhanced_imu, gate_emg, gate_imu = self.cross_residual(seq_emg, seq_imu, feat_emg, feat_imu)
        logit_emg_enh = self.emg_encoder.head(enhanced_emg)
        logit_imu_enh = self.imu_encoder.head(enhanced_imu)
        residual_emg = enhanced_emg - feat_emg
        residual_imu = enhanced_imu - feat_imu
        align_emg = F.normalize(self.emg_align_proj(feat_emg), dim=1)
        align_imu = F.normalize(self.imu_align_proj(feat_imu), dim=1)
        align_emg_enh = F.normalize(self.emg_align_proj(enhanced_emg), dim=1)
        align_imu_enh = F.normalize(self.imu_align_proj(enhanced_imu), dim=1)
        raw_feat_emg = feat_emg
        raw_feat_imu = feat_imu
        raw_residual_emg = residual_emg
        raw_residual_imu = residual_imu

        if self.training and self.modality_dropout_p > 0:
            drop_emg = (torch.rand(feat_emg.size(0), 1, device=feat_emg.device) < (self.modality_dropout_p / 2)).float()
            drop_imu = (torch.rand(feat_imu.size(0), 1, device=feat_imu.device) < (self.modality_dropout_p / 2)).float()
            both_dropped = drop_emg * drop_imu
            drop_emg = drop_emg * (1.0 - both_dropped)
            drop_imu = drop_imu * (1.0 - both_dropped)
            keep_emg = 1.0 - drop_emg
            keep_imu = 1.0 - drop_imu
            feat_emg = feat_emg * keep_emg
            residual_emg = residual_emg * keep_emg
            feat_imu = feat_imu * keep_imu
            residual_imu = residual_imu * keep_imu

        fused_feat = torch.cat([feat_emg, feat_imu, residual_emg, residual_imu], dim=1)
        fusion_logits = self.fusion_mlp(fused_feat)
        return {
            "fusion_logits": fusion_logits,
            "emg_logits": logit_emg,
            "imu_logits": logit_imu,
            "emg_enh_logits": logit_emg_enh,
            "imu_enh_logits": logit_imu_enh,
            "feat_emg": raw_feat_emg,
            "feat_imu": raw_feat_imu,
            "enhanced_emg": enhanced_emg,
            "enhanced_imu": enhanced_imu,
            "residual_emg": raw_residual_emg,
            "residual_imu": raw_residual_imu,
            "align_emg": align_emg,
            "align_imu": align_imu,
            "align_emg_enh": align_emg_enh,
            "align_imu_enh": align_imu_enh,
            "gate_emg": gate_emg.mean(),
            "gate_imu": gate_imu.mean(),
            "gate_emg_map": gate_emg,
            "gate_imu_map": gate_imu,
        }
