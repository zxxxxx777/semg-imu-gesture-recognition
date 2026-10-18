import os
import torch


# ==========================================
# 1. EPN-107 200 Hz IMU configuration
# ==========================================
DATA_ROOT = "/hdd/zhaozixuan/EMG-EPN-107"
OUTER_TRAIN_GROUP = "training"

# 200 Hz users kept identical to the fusion baseline for a fair comparison.
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

# Pretrain uses the internal training split from source 200 Hz users.
PRETRAIN_SECTIONS = ("training",)

# Target fine-tuning/evaluation follow the DB7-style repetition split on
# the target user's internal training split.
TARGET_TRAIN_SECTIONS = ("training",)
TARGET_TEST_SECTIONS = ("training",)

TRAIN_REPS = [1, 2, 3]
TEST_REPS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

SAVE_ROOT = "/hdd/zhaozixuan/baseline_EPN/result"
SAVE_DIR = os.path.join(SAVE_ROOT, f"IMU_Finetune_{TARGET_USER}")


# ==========================================
# 2. Shared training configuration
# ==========================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 2026
DETERMINISTIC = True

# EPN 200 Hz: 40 samples ~= 200 ms. Use non-overlapping windows to reduce
# strong correlation between adjacent EPN trial windows.
WINDOW_SIZE = 40
STEP_SIZE = 40

NUM_CLASSES = 11
EMG_CHANNELS = 8
IMU_CHANNELS = 10  # gyro(3) + accel(3) + quaternions(4)
EPN_SAMPLING_RATE = 200
EPN_REPS_PER_CLASS = 15
EPN_RELAX_TRIALS = 15

DROP_MIXED_WINDOW = True
PRETRAIN_BATCH_SIZE = 128
FINETUNE_BATCH_SIZE = 32
BATCH_SIZE = PRETRAIN_BATCH_SIZE
EPOCHS = 100
LR = 1e-3
FINETUNE_LR = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.3
DROPOUT_CNN = 0.3
DROPOUT_HEAD = 0.5
DROPOUT_LSTM = 0.3
EARLY_STOP_PATIENCE = 30
GRAD_CLIP = 1.0
