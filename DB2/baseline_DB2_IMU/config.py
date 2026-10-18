import os

try:
    import torch
except ImportError:
    torch = None


# ==========================================
# 1. 单受试者 (Intra-subject) 配置
# ==========================================
idx = 10
LOSO_BASE_DATA_DIR = "/hdd/zhaozixuan/DB2"

# DB2 动作集:
# E1: labels 1-17, E2: labels 18-40, E3: labels 41-49
# 当前只使用 E1+E2 的 40 类动作。
EXERCISES = [1, 2]
NUM_CLASSES = 40

# 如果要使用 DB2 全部 49 类动作，改为:
# EXERCISES = [1, 2, 3]
# NUM_CLASSES = 49

DATA_DIR = f"{LOSO_BASE_DATA_DIR}/Subject_{idx}"
FILE_LIST = [f"S{idx}_E{exercise}_A1.mat" for exercise in EXERCISES]

SAVE_DIR = f"/hdd/zhaozixuan/baseline_DB2_IMU/result/sub{idx}"

TRAIN_REPS = [1, 3, 4, 6]
TEST_REPS = [2, 5]
# TRAIN_REPS = [1,3]
# TEST_REPS = [2,4,5,6]


# ==========================================
# 2. 跨受试者 (Inter-subject/LOSO) 配置区
# ==========================================
LOSO_ALL_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LOSO_TEST_SUBJECT = [10]
# LOSO_TEST_SUBJECT = [9, 10]
LOSO_TRAIN_SUBJECTS = [s for s in LOSO_ALL_SUBJECTS if s != LOSO_TEST_SUBJECT]

LOSO_SAVE_DIR = f"/hdd/zhaozixuan/baseline_DB2_IMU/result/LOSO_sub{LOSO_TEST_SUBJECT}"


# ==========================================
# 3. 全局共享超参数与配置 (单人/跨人 通用)
# ==========================================
DEVICE = "cuda:1" if torch is not None and torch.cuda.is_available() else "cpu"

RANDOM_SEED = 2026
DETERMINISTIC = True

WINDOW_SIZE = 400
STEP_SIZE = 200

ZC_THRESHOLD = 1e-6
SSC_THRESHOLD = 1e-6

MAT_MOTION_KEY = "acc"
MAT_LABEL_KEY = "restimulus"
MAT_REP_KEY = "rerepetition"

DROP_MIXED_WINDOW = True

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.3
DROPOUT_CNN = 0.3
DROPOUT_HEAD = 0.5
DROPOUT_LSTM = 0.3
EARLY_STOP_PATIENCE = 30
GRAD_CLIP = 1.0
