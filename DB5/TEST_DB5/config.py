import torch

# ==========================================
# 1. 基础环境与网络参数
# ==========================================
# 设备选择：CUDA 可用时默认使用 cuda:1
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# 核心网络超参数
LSTM_HIDDEN = 128
DROPOUT_CNN = 0.3
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

# NinaPro DB5 .mat 文件中的键名映射
MAT_EMG_KEY = 'emg'
MAT_MOTION_KEY = 'acc'          # DB5 的运动传感器为 3 轴加速度
MAT_LABEL_KEY = 'restimulus'
MAT_REP_KEY = 'rerepetition'

# ==========================================
# 3. 跨受试者 (Inter-subject/LOSO) 数据集路径配置
# ==========================================
# 数据集根目录
LOSO_BASE_DATA_DIR = "/hdd/zhaozixuan/DB5"

# DB5 动作集: E1=A(12类), E2=B(17类), E3=C(23类)
EXERCISES = [2, 3]  # 使用 B+C 动作集
# EXERCISES = [1, 2, 3]  # 使用 A+B+C 动作集

NUM_CLASSES = 40  # B+C: 17 + 23
# NUM_CLASSES = 52  # A+B+C: 12 + 17 + 23

DB5_EXERCISE_CLASS_COUNTS = {
    1: 12,
    2: 17,
    3: 23,
}

EXERCISE_LABEL_OFFSETS = {}
_label_offset = 0
for _exercise in EXERCISES:
    EXERCISE_LABEL_OFFSETS[_exercise] = _label_offset
    _label_offset += DB5_EXERCISE_CLASS_COUNTS[_exercise]

# 跨人预训练被试集合
PRETRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 目标被试 (用于微调和最终测试)
TARGET_SUBJECT = 10

# 目标被试 (Sub 10) 的 Repetition 划分
# Rep 1 用于微调，Rep 2–6 用于测试
TRAIN_REPS = [1]
TEST_REPS = [2, 3, 4, 5, 6]

# ==========================================
# 4. 单受试者 (Intra-subject) 兼容配置
# ==========================================
# 供 utils.py 中的 load_data 函数使用
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
EARLY_STOP_PATIENCE = 30     # 早停耐心值，与 TEST_DB7 保持一致

# 6. 损失权重
PRETRAIN_ALPHA = 0.3
FINETUNE_ALPHA = 0.3
PRETRAIN_BETA_ENH = 0.1
PRETRAIN_BETA_KL = 0.02
PRETRAIN_BETA_CONS = 0.02
PRETRAIN_BETA_GATE = 0.001
PRETRAIN_BETA_REL_ALIGN = 0.0

FINETUNE_BETA_ENH = 0.2
FINETUNE_BETA_KL = 0.03
FINETUNE_BETA_CONS = 0.03
FINETUNE_BETA_GATE = 0.01
FINETUNE_BETA_REL_ALIGN = 0.07

# 7. 双向残差交叉注意力互学习参数
FUSION_METHOD_NAME = "BiResidualCrossAttentionMutualLearning"
CROSS_ATTN_HEADS = 4
CROSS_ATTN_DROPOUT = 0.1
MAX_CROSS_RES_SCALE = 0.2
MUTUAL_KL_TEMPERATURE = 1.0
MODALITY_DROPOUT_P = 0.1
ALIGN_PROJ_DIM = 64

# 8. 实验开关
# 默认使用固定损失权重；同方差不确定性权重用于可选消融实验。
USE_HOMOSCEDASTIC_LOSS = False

# 9. 预训练缓存
# 启用缓存时，可复用预训练权重和归一化参数，跳过 Phase 1。
USE_PRETRAIN_CACHE = False
FORCE_PRETRAIN = False
PRETRAIN_CACHE_NAME = "pretrain_bires_crossattn_mutual_moddrop_gatesparse_relalign_db5_v1.pth"
