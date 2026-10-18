import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# 注意力池化层，对时间序列数据进行加权池化。让模型自动学习输入序列中不同时间步的重要性，然后根据这些重要性权重对特征进行加权求和。
class AttnPool(nn.Module):
    """Attention pooling over time: (B,T,F) -> (B,F)""" # B是批量大小，T是时间步数，F是特征维度
    def __init__(self, feat_dim: int):   # feat_dim是输入特征维度
        super().__init__()   # 调用父类
        self.w = nn.Linear(feat_dim, 1) # 创建一个线性层 self.w ，将输入特征映射到 1 维

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,T,F)，是LSTM的输出，包含每个时间步的特征表示
        a = self.w(h)            
        # 计算注意力分数,将每个时间步的特征向量 (F) 映射到一个标量值,(B,T,1)
        a = torch.softmax(a, dim=1)    # 归一化注意力分数,形成概率分布,(B,T,1)
        return (a * h).sum(dim=1)      # 将注意力权重与对应时间步的特征相乘，在时间维度上求和。得到每个样本的加权特征表示,(B,F)

class EMGHandNet1D(nn.Module): 
    """
    EMGHandNet-style classifier for EMG windows.
    Input:
      x: (B, C=12, L=2000)  (you can use any L, e.g., 2000)
    Output:
      features: (B, D)      embedding for FID-like use
      logits:   (B, K)      class logits for classification/IS-like use
    """
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

        # CNN feature extractor: (B,C,L) -> (B,256,T) 
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

        # BiLSTM over time: (B,T,256) -> (B,T,2H)  
        self.lstm = nn.LSTM(   
            input_size=256,
            hidden_size=lstm_hidden,  
            num_layers=lstm_layers,  
            batch_first=True,      
            bidirectional=True,    
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
        )

        # 注意力池化层,Attention pooling: (B,T,2H) -> (B,2H)
        self.pool = AttnPool(feat_dim=2 * lstm_hidden)   

        # Head: (B,2H) -> (B,K)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_head),
            nn.Linear(256, num_classes), 
        )
        
    def forward(self, x: torch.Tensor):   
        z = self.cnn(x)         # (B,256,T)
        z = z.transpose(1, 2)   # (B,T,256)
        h, _ = self.lstm(z)    
        features = self.pool(h) # (B,2H)
        logits = self.head(features)
        
        # features shape: (B, 256) 
        return features, logits


# ==========================================
# 特征级多模态融合网络 (Feature-Level Fusion + MLP)
# ==========================================
class FeatureFusionNet(nn.Module):
    def __init__(self, emg_ch: int, imu_ch: int, num_classes: int = 40, lstm_hidden: int = 128, dropout_head: float = DROPOUT_HEAD):
        super().__init__()
        
        # 实例化两个独立的特征提取器 (共享相同的超参数结构)
        self.emg_encoder = EMGHandNet1D(in_ch=emg_ch, num_classes=num_classes, lstm_hidden=lstm_hidden)
        self.imu_encoder = EMGHandNet1D(in_ch=imu_ch, num_classes=num_classes, lstm_hidden=lstm_hidden)

        # EMG和IMU经过 pool() 后的特征维度均为 2 * lstm_hidden (默认 = 256)
        # 拼接后维度：256 + 256 = 512
        concat_dim = (2 * lstm_hidden) * 2

        # MLP 融合分类头
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),      # 加入 BN 增加特征融合后的训练稳定性
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_head),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_emg, x_imu):
        # 1. 提取各自的特征 (B, 256) 
        # 这里同时也返回各自分支的 logits，可以用于计算辅助损失(Auxiliary Loss)，加速编码器收敛
        feat_emg, logit_emg = self.emg_encoder(x_emg)
        feat_imu, logit_imu = self.imu_encoder(x_imu)

        # 2. 特征级拼接 (B, 512)
        fused_feat = torch.cat([feat_emg, feat_imu], dim=1)

        # 3. 经过 MLP 得到最终的融合分类结果
        fusion_logits = self.fusion_mlp(fused_feat)

        return fusion_logits, logit_emg, logit_imu
