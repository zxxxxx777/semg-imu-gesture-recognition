import os
import torch


# ==========================================
# 1. EPN-107 200 Hz SEMG + IMU data config
# ==========================================
DATA_ROOT = "/hdd/zhaozixuan/EMG-EPN-107"
OUTER_TRAIN_GROUP = "training"

EPN_200HZ_USERS = [
    "user_001",
    "user_024",
    "user_025",
    "user_026",
    "user_027",
    "user_032",
    "user_033",
    "user_035",
    "user_036",
    "user_037",
    "user_038",
    "user_039",
    "user_040",
    "user_041",
    "user_042",
    "user_043",
    "user_045",
    "user_047",
    "user_049",
]

TARGET_USER = "user_049"
PRETRAIN_USERS = [
    "user_001",
    "user_024",
    "user_025",
    "user_026",
    "user_027",
    "user_032",
    "user_033",
    "user_035",
    "user_036",
]

PRETRAIN_SECTIONS = ("training",)
TARGET_TRAIN_SECTIONS = ("training",)
TARGET_TEST_SECTIONS = ("training",)

TRAIN_REPS = [1, 2, 3]
TEST_REPS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

SAVE_ROOT = "/hdd/zhaozixuan/TEST_EPN/result"
SAVE_DIR = os.path.join(SAVE_ROOT, f"Fusion_BiResCrossAttn_{TARGET_USER}")


# ==========================================
# 2. Shared model/training config
# ==========================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 2026
DETERMINISTIC = True

LSTM_HIDDEN = 128
DROPOUT_CNN = 0.3
DROPOUT_LSTM = 0.3
DROPOUT_HEAD = 0.5

WINDOW_SIZE = 40
PRETRAIN_STEP_SIZE = 40
FINETUNE_STEP_SIZE = 40
STEP_SIZE = PRETRAIN_STEP_SIZE
DROP_MIXED_WINDOW = True
PRETRAIN_BATCH_SIZE = 128
FINETUNE_BATCH_SIZE = 32
BATCH_SIZE = PRETRAIN_BATCH_SIZE

NUM_CLASSES = 11
EMG_CHANNELS = 8
IMU_CHANNELS = 10
EPN_SAMPLING_RATE = 200
EPN_REPS_PER_CLASS = 15
EPN_RELAX_TRIALS = 15

PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 1e-3

EPOCHS = 100
FINETUNE_LR = 2e-4
WEIGHT_DECAY = 1e-4
PRETRAIN_LABEL_SMOOTHING = 0.3
FINETUNE_LABEL_SMOOTHING = 0.3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 30


# ==========================================
# 3. Mutual-learning fusion loss weights
# ==========================================
PRETRAIN_ALPHA = 0.1
FINETUNE_ALPHA = 0.05
PRETRAIN_BETA_ENH = 0.05
PRETRAIN_BETA_KL = 0.01
PRETRAIN_BETA_CONS = 0.01
PRETRAIN_BETA_GATE = 0.0
PRETRAIN_BETA_REL_ALIGN = 0.0

FINETUNE_BETA_ENH = 0.05
FINETUNE_BETA_KL = 0.0
FINETUNE_BETA_CONS = 0.01
FINETUNE_BETA_GATE = 0.0
FINETUNE_BETA_REL_ALIGN = 0.0


# ==========================================
# 4. Bi-residual cross-attention params
# ==========================================
FUSION_METHOD_NAME = "EPN_BiResidualCrossAttentionMutualLearning"
CROSS_ATTN_HEADS = 4
CROSS_ATTN_DROPOUT = 0.1
MAX_CROSS_RES_SCALE = 0.2
MUTUAL_KL_TEMPERATURE = 2.0
MODALITY_DROPOUT_P = 0.1
ALIGN_PROJ_DIM = 64


# ==========================================
# 5. Pretrain cache
# ==========================================
USE_PRETRAIN_CACHE = False
FORCE_PRETRAIN = True
PRETRAIN_CACHE_NAME = "pretrain_epn_bires_crossattn_mutual_rep1to3_v1.pth"
