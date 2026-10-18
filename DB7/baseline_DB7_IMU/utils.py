import os
import numpy as np
import scipy.io as scio
from config import *


# ==========================================
# 单受试者 (Intra-subject) 专用数据加载
# ==========================================
def load_data():
    emg_list, label_list, rep_list = [], [], []

    for file in FILE_LIST:
        data = scio.loadmat(os.path.join(DATA_DIR, file))
        emg_list.append(data[MAT_EMG_KEY])
        label_list.append(data[MAT_LABEL_KEY])
        rep_list.append(data[MAT_REP_KEY])

    emg = np.vstack(emg_list)     # 垂直堆叠,如(200,12)与(300,12)堆叠后为(500,12)
    label = np.vstack(label_list).squeeze() # squeeze()函数用于去除数组中维度为1的轴，使得标签成为一维数组
    rep = np.vstack(rep_list).squeeze()

    return emg, label, rep


# ==========================================
# 跨受试者 (Inter-subject/LOSO) 专用数据加载
# ==========================================
def load_single_subject_emg(subject_id):
    """
    加载指定受试者的连续 sEMG 数据，并过滤静息状态和无效重复。
    供 LOSO (Leave-One-Subject-Out) 跨人验证使用。
    """
    # 注意这里使用了新的配置变量 LOSO_BASE_DATA_DIR
    data_dir = f"{LOSO_BASE_DATA_DIR}/Subject_{subject_id}"
    #file_list = [f"S{subject_id}_E1_A1.mat"]#只使用B动作集
    file_list = [f"S{subject_id}_E1_A1.mat", f"S{subject_id}_E2_A1.mat"]  #使用B动作集和C动作集
    
    emg_list, label_list, rep_list = [], [], []

    for file in file_list:
        data = scio.loadmat(os.path.join(data_dir, file))
        emg_list.append(data[MAT_EMG_KEY])
        label_list.append(data[MAT_LABEL_KEY])
        rep_list.append(data[MAT_REP_KEY])

    emg = np.vstack(emg_list)
    label = np.vstack(label_list).squeeze()
    rep = np.vstack(rep_list).squeeze()

    # 严格过滤静息类别(0)和无效重复(通常rep=0代表未标记或练习阶段)
    idx = np.where((label != 0) & (rep != 0))[0]
    
    emg = emg[idx]
    label = label[idx] - 1  # 标签从1-40映射为0-39，适应PyTorch交叉熵损失要求
    
    return emg, label

def load_single_subject_imu(subject_id):
    """
    加载指定受试者的连续 IMU 数据（Acc+Gyro+Mag），并过滤静息状态。
    供 LOSO (Leave-One-Subject-Out) 跨人验证使用。
    """
    # 动态加载对应受试者的数据目录
    data_dir = f"{LOSO_BASE_DATA_DIR}/Subject_{subject_id}"
    #file_list = [f"S{subject_id}_E1_A1.mat"]#只使用B动作集
    file_list = [f"S{subject_id}_E1_A1.mat", f"S{subject_id}_E2_A1.mat"]  #使用B动作集和C动作集
    
    acc_list, gyro_list, mag_list, label_list, rep_list = [], [], [], [], []

    for file in file_list:
        data = scio.loadmat(os.path.join(data_dir, file))
        acc_list.append(data['acc'])
        gyro_list.append(data['gyro'])
        mag_list.append(data['mag'])
        label_list.append(data[MAT_LABEL_KEY])
        rep_list.append(data[MAT_REP_KEY])

    acc = np.vstack(acc_list)
    gyro = np.vstack(gyro_list)
    mag = np.vstack(mag_list)
    label = np.vstack(label_list).squeeze()
    rep = np.vstack(rep_list).squeeze()

    # 将 3轴加速度、3轴角速度、3轴磁力计在通道维度上拼接 (最终为 12x9=108 维)
    imu = np.concatenate([acc, gyro, mag], axis=1)

    # 严格过滤静息类别(0)和无效重复
    idx = np.where((label != 0) & (rep != 0))[0]
    
    imu = imu[idx]
    label = label[idx] - 1  # 标签从1-40映射为0-39
    
    return imu, label
# ==========================================
# 全局共享处理函数 (划分、归一化、滑窗)
# ==========================================
def split_dataset(emg, label, rep):
    idx = np.where((label != 0) & (rep != 0))[0] # 去除标签为0且重复为0的数据
    # np.where()函数返回索引,使用[0]提取索引数组而非元组

    emg = emg[idx]
    label = label[idx] - 1  # 标签从1-40为0-39,适应PyTorch的交叉熵损失函数要求标签从0开始
    rep = rep[idx]

    train_idx = np.isin(rep, TRAIN_REPS) # isin()判断rep中的元素是否在TRAIN_REPS中,返回索引的数组
    test_idx = np.isin(rep, TEST_REPS)

    return emg[train_idx], label[train_idx], emg[test_idx], label[test_idx]


def zscore_norm(train, test):   # 只用训练集的均值和标准差来归一化,不使用测试集数据以避免数据泄漏
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8  # 加上一个小常数以避免除以零的情况

    return (train - mean) / std, (test - mean) / std


def sliding_window(X, y):
    Xw, yw = [], []

    total = (len(X) - WINDOW_SIZE) // STEP_SIZE + 1 # 滑动窗口总数
    drop = 0

    for i in range(total):   # 遍历窗口
        s = i * STEP_SIZE    # 窗口起始索引
        e = s + WINDOW_SIZE  # 窗口结束索引

        wx, wy = X[s:e], y[s:e]
       
        # DROP_MIXED_WINDOW表示启动标签一致性检测，如果wy中的所有元素都相等np.all返回True
        if DROP_MIXED_WINDOW and not np.all(wy == wy[0]):  
            drop += 1
            continue

        label = wy[0] if DROP_MIXED_WINDOW else np.bincount(wy).argmax()
        # 如果丢弃混合窗口则取第一个标签。否则用多数投票选择最常见的标签。
        # bincount()统计每个标签的出现次数，argmax()返回出现次数最多的标签索引

        Xw.append(wx)
        yw.append(label)

    print(f"窗口总数:{total}, 保留:{len(Xw)}, 丢弃:{drop}")

    return np.array(Xw), np.array(yw)