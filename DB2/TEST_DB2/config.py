import os

try:
    import torch
except ImportError:
    torch = None

# ==========================================
# 1. 基础环境与网络参数
# ==========================================
# 设备选择：默认使用你报错中显示的 cuda:1
DEVICE = 'cuda:1' if torch is not None and torch.cuda.is_available() else 'cpu'
SEED = 2026

# 核心网络超参数 
LSTM_HIDDEN = 128
DROPOUT_CNN = 0.3   #DROPOUT的含义是将神经元设为0的概率，防止过拟合
DROPOUT_LSTM = 0.3
DROPOUT_HEAD = 0.5 

# ==========================================
# 2. 数据集与滑动窗口参数
# ==========================================
WINDOW_SIZE = 400        # 滑动窗口的长度
PRETRAIN_STEP_SIZE = 200 # 预训练步长
FINETUNE_STEP_SIZE = 200 # 微调/测试步长
STEP_SIZE = PRETRAIN_STEP_SIZE
DROP_MIXED_WINDOW = True # 是否丢弃跨越两个动作（包含多种标签）的混合窗口
BATCH_SIZE = 128         # 批次大小

# NinaPro DB2 .mat 文件中的键名映射
MAT_EMG_KEY = 'emg'
MAT_MOTION_KEY = 'acc'          # DB2 只有 Trigno 加速度计数据: 12 electrodes x 3 axes = 36 channels
MAT_LABEL_KEY = 'restimulus'
MAT_REP_KEY = 'rerepetition'

# ==========================================
# 3. 跨受试者 (Inter-subject/LOSO) 数据集路径配置
# ==========================================
# 你的基础数据根目录
LOSO_BASE_DATA_DIR = "/hdd/zhaozixuan/DB2"

# DB2 动作集:
# E1: labels 1-17, E2: labels 18-40, E3: labels 41-49
# 当前只使用 E1+E2 的 40 类动作。
EXERCISES = [1, 2]
NUM_CLASSES = 40

# 如果要使用 DB2 全部 49 类动作，改为:
# EXERCISES = [1, 2, 3]
# NUM_CLASSES = 49

# 跨人预训练被试集合
PRETRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 目标被试 (用于微调和最终测试)
TARGET_SUBJECT = 10

# 目标被试 (Sub 10) 的 Repetition 划分
# 根据你的实际需求，Rep 1 用于微调，Rep 2-6 或 2,5 用于测试
TRAIN_REPS = [1]
TEST_REPS = [2, 3, 4, 5, 6]

# ==========================================
# 4. 单受试者 (Intra-subject) 兼容配置保留区
# ==========================================
# (这部分主要为了兼容你之前的 utils.py 中的 load_data 函数)
idx = TARGET_SUBJECT
DATA_DIR = f"{LOSO_BASE_DATA_DIR}/Subject_{idx}"
FILE_LIST = [f"S{idx}_E{exercise}_A1.mat" for exercise in EXERCISES]

# ==========================================
# 5. 训练与优化器参数
# ==========================================
# 预训练阶段 (Phase 1)
PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 1e-3

# 微调阶段 (Phase 2)
EPOCHS = 100
FINETUNE_LR = 1e-4
ENCODER_LR_MULT = 1.0
GATE_LR_MULT = 1.0
FINETUNE_WEIGHT_DECAY = 1e-4

# 通用优化参数
WEIGHT_DECAY = 1e-4         # 权重衰减系数，防止过拟合
PRETRAIN_LABEL_SMOOTHING = 0.3
FINETUNE_LABEL_SMOOTHING = 0.3
GRAD_CLIP = 1.0              # 梯度裁剪阈值，防止 LSTM 梯度爆炸
EARLY_STOP_PATIENCE = 30     # 早停耐心值

# 6. 损失权重
PRETRAIN_ALPHA = 0.3
FINETUNE_ALPHA = 0.2
PRETRAIN_BETA_ENH = 0.05
PRETRAIN_BETA_KL = 0.02
PRETRAIN_BETA_CONS = 0.02
PRETRAIN_BETA_GATE = 0.001
PRETRAIN_BETA_REL_ALIGN = 0.001

FINETUNE_BETA_ENH = 0.1
FINETUNE_BETA_KL = 0.015
#
FINETUNE_BETA_CONS = 0.03
FINETUNE_BETA_GATE = 0.001
FINETUNE_BETA_REL_ALIGN = 0.005

# 7. 双向残差交叉注意力互学习参数
FUSION_METHOD_NAME = "BiResidualCrossAttentionMutualLearning"
CROSS_ATTN_HEADS = 4
CROSS_ATTN_DROPOUT = 0.1
MAX_CROSS_RES_SCALE = 0.3
MUTUAL_KL_TEMPERATURE = 2.0
MODALITY_DROPOUT_P = 0.1
ALIGN_PROJ_DIM = 64

# 8. 预训练缓存
USE_PRETRAIN_CACHE = True
FORCE_PRETRAIN = False
PRETRAIN_CACHE_NAME = "pretrain_db2_bires_crossattn_mutual_moddrop_gatesparse_relalign_v1.pth"
