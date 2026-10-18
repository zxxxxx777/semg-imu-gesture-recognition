import torch
import os

# ==========================================
# 1. 单受试者 (Intra-subject) 配置
# ==========================================
idx = 10
DATA_DIR = f"/hdd/zhaozixuan/DB7/Subject_{idx}"

#E1_A1.mat是B动作集，E2_A1.mat是C动作集
FILE_LIST = [
    f"S{idx}_E1_A1.mat",
    f"S{idx}_E2_A1.mat"
]

SAVE_DIR = f"/hdd/zhaozixuan/baseline/result/sub{idx}"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_REPS = [1, 3, 4, 6]
TEST_REPS = [2, 5]
# TRAIN_REPS = [1,3]
# TEST_REPS = [2,4,5,6]


# ==========================================
# 2. 跨受试者 (Inter-subject/LOSO) 配置区
# ==========================================
LOSO_BASE_DATA_DIR = "/hdd/zhaozixuan/DB7"

# 跨受试者实验的受试者划分
LOSO_ALL_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8,9]
LOSO_TEST_SUBJECT = [10]  
#LOSO_TEST_SUBJECT = [9, 10]  
LOSO_TRAIN_SUBJECTS = [s for s in LOSO_ALL_SUBJECTS if s != LOSO_TEST_SUBJECT]

# 使用专门的保存路径和变量名，避免与上方单人模式冲突
LOSO_SAVE_DIR = f"/hdd/zhaozixuan/baseline/result/LOSO_sub{LOSO_TEST_SUBJECT}"
os.makedirs(LOSO_SAVE_DIR, exist_ok=True)


# ==========================================
# 3. 全局共享超参数与配置 (单人/跨人 通用)
# ==========================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# 固定随机数，保证实验尽量可复现
RANDOM_SEED = 2026
DETERMINISTIC = True

WINDOW_SIZE = 400
STEP_SIZE = 200

ZC_THRESHOLD = 1e-6
SSC_THRESHOLD = 1e-6

MAT_EMG_KEY = "emg"
MAT_LABEL_KEY = "restimulus"
MAT_REP_KEY = "rerepetition"

NUM_CLASSES = 40
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
