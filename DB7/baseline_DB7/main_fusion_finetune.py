# 特征级多模态融合+MLP 两阶段训练：DB7 1-20 号受试者五折预训练 + 单受试者独立微调
import sys
import os
import random
import csv
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import scipy.io as scio

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from models import FeatureFusionNet # 修改点：导入新的特征融合网络
from train import EarlyStopping

ALL_SUBJECTS = list(range(1, 21))
FOLD_TARGET_SUBJECTS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]
FINETUNE_TRAIN_REPS = [1]
FINETUNE_TEST_REPS = [2, 3, 4, 5, 6]
FINETUNE_LR = 1e-4  

RESULT_ROOT = "/hdd/zhaozixuan/baseline/result/DB7_5fold_Fusion_Feature_Finetune"
SUMMARY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DB7_5fold_Fusion_Feature_Finetune_summary.csv"
)

def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as exc:
            print(f"[Seed] 无法启用完全确定性算法: {exc}")

def seed_worker(worker_id):
    worker_seed = (RANDOM_SEED + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def make_loader(dataset, batch_size, shuffle, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, X_emg, X_imu, y):
        self.X_emg = torch.tensor(X_emg, dtype=torch.float32)
        self.X_imu = torch.tensor(X_imu, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X_emg)

    def __getitem__(self, idx):
        x_emg = self.X_emg[idx].transpose(0, 1)
        x_imu = self.X_imu[idx].transpose(0, 1)
        return x_emg, x_imu, self.y[idx]

def train_fusion_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_emg, x_imu, y in loader:
        x_emg, x_imu, y = x_emg.to(device), x_imu.to(device), y.to(device)
            
        optimizer.zero_grad()
        
        # 修改点：直接通过融合模型前向传播，得到最终融合 logits
        fusion_logits, _, _ = model(x_emg, x_imu)
        
        loss = criterion(fusion_logits, y)
        
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_fusion(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x_emg, x_imu, y in loader:
            x_emg, x_imu, y = x_emg.to(device), x_imu.to(device), y.to(device)
            
            # 修改点：测试时同样直接调用前向传播
            fusion_logits, _, _ = model(x_emg, x_imu)
            loss = criterion(fusion_logits, y) # 测试时只关注融合主损失
            
            total_loss += loss.item()
            pred = fusion_logits.argmax(1)
            correct += (pred == y).sum().item()
    
    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc

def sliding_window_multimodal(X_emg, X_imu, y):
    X_emg_w, X_imu_w, y_w = [], [], []
    total = (len(X_emg) - WINDOW_SIZE) // STEP_SIZE + 1
    drop = 0
    for i in range(total):
        s = i * STEP_SIZE
        e = s + WINDOW_SIZE
        wy = y[s:e]
        if DROP_MIXED_WINDOW and not np.all(wy == wy[0]):
            drop += 1
            continue
        label = wy[0] if DROP_MIXED_WINDOW else np.bincount(wy).argmax()
        X_emg_w.append(X_emg[s:e])
        X_imu_w.append(X_imu[s:e])
        y_w.append(label)
    return np.array(X_emg_w), np.array(X_imu_w), np.array(y_w)

def sliding_window_multimodal_by_segment(X_emg, X_imu, y, rep):
    X_emg_w, X_imu_w, y_w = [], [], []
    if len(y) < WINDOW_SIZE:
        return (
            np.empty((0, WINDOW_SIZE, X_emg.shape[1]), dtype=X_emg.dtype),
            np.empty((0, WINDOW_SIZE, X_imu.shape[1]), dtype=X_imu.dtype),
            np.empty((0,), dtype=y.dtype)
        )

    change_points = np.where((y[1:] != y[:-1]) | (rep[1:] != rep[:-1]))[0] + 1
    boundaries = np.concatenate(([0], change_points, [len(y)]))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < WINDOW_SIZE:
            continue
        Xw_emg, Xw_imu, Yw = sliding_window_multimodal(
            X_emg[start:end],
            X_imu[start:end],
            y[start:end]
        )
        if len(Yw) == 0:
            continue
        X_emg_w.append(Xw_emg)
        X_imu_w.append(Xw_imu)
        y_w.append(Yw)

    if not y_w:
        return (
            np.empty((0, WINDOW_SIZE, X_emg.shape[1]), dtype=X_emg.dtype),
            np.empty((0, WINDOW_SIZE, X_imu.shape[1]), dtype=X_imu.dtype),
            np.empty((0,), dtype=y.dtype)
        )

    return np.concatenate(X_emg_w, axis=0), np.concatenate(X_imu_w, axis=0), np.concatenate(y_w, axis=0)

def load_subject_multimodal_with_rep(subject_id):
    data_dir = f"{LOSO_BASE_DATA_DIR}/Subject_{subject_id}"
    file_list = [f"S{subject_id}_E1_A1.mat", f"S{subject_id}_E2_A1.mat"]
    
    emg_list, acc_list, gyro_list, mag_list, label_list, rep_list = [], [], [], [], [], []

    for file in file_list:
        data = scio.loadmat(os.path.join(data_dir, file))
        emg_list.append(data[MAT_EMG_KEY])
        acc_list.append(data['acc'])
        gyro_list.append(data['gyro'])
        mag_list.append(data['mag'])
        label_list.append(data[MAT_LABEL_KEY])
        rep_list.append(data[MAT_REP_KEY])

    emg = np.vstack(emg_list)
    acc = np.vstack(acc_list)
    gyro = np.vstack(gyro_list)
    mag = np.vstack(mag_list)
    imu = np.concatenate([acc, gyro, mag], axis=1)

    label = np.vstack(label_list).squeeze()
    rep = np.vstack(rep_list).squeeze()

    idx = np.where((label != 0) & (rep != 0))[0]
    return emg[idx], imu[idx], label[idx] - 1, rep[idx]


def build_pretrain_loader(pretrain_subjects):
    pretrain_emg_list, pretrain_imu_list, pretrain_label_list, pretrain_rep_list = [], [], [], []

    for sub_id in pretrain_subjects:
        emg, imu, label, rep = load_subject_multimodal_with_rep(sub_id)
        pretrain_emg_list.append(emg)
        pretrain_imu_list.append(imu)
        pretrain_label_list.append(label)
        pretrain_rep_list.append(rep)

    global_pretrain_emg = np.vstack(pretrain_emg_list)
    emg_mean = global_pretrain_emg.mean(axis=0)
    emg_std = global_pretrain_emg.std(axis=0) + 1e-8
    del global_pretrain_emg

    global_pretrain_imu = np.vstack(pretrain_imu_list)
    imu_mean = global_pretrain_imu.mean(axis=0)
    imu_std = global_pretrain_imu.std(axis=0) + 1e-8
    del global_pretrain_imu

    X_emg_pretrain_w, X_imu_pretrain_w, y_pretrain_w = [], [], []
    for i in range(len(pretrain_emg_list)):
        norm_emg = (pretrain_emg_list[i] - emg_mean) / emg_std
        norm_imu = (pretrain_imu_list[i] - imu_mean) / imu_std
        
        Xw_emg, Xw_imu, yw = sliding_window_multimodal_by_segment(
            norm_emg,
            norm_imu,
            pretrain_label_list[i],
            pretrain_rep_list[i]
        )
        X_emg_pretrain_w.append(Xw_emg)
        X_imu_pretrain_w.append(Xw_imu)
        y_pretrain_w.append(yw)

    X_emg_pretrain = np.concatenate(X_emg_pretrain_w, axis=0)
    X_imu_pretrain = np.concatenate(X_imu_pretrain_w, axis=0)
    y_pretrain = np.concatenate(y_pretrain_w, axis=0)
    del pretrain_emg_list, pretrain_imu_list, pretrain_label_list, pretrain_rep_list
    del X_emg_pretrain_w, X_imu_pretrain_w, y_pretrain_w

    pretrain_loader = make_loader(
        MultiModalDataset(X_emg_pretrain, X_imu_pretrain, y_pretrain),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    emg_ch = X_emg_pretrain.shape[2]
    imu_ch = X_imu_pretrain.shape[2]
    sample_count = len(y_pretrain)

    del X_emg_pretrain, X_imu_pretrain, y_pretrain

    return pretrain_loader, emg_mean, emg_std, imu_mean, imu_std, emg_ch, imu_ch, sample_count


def build_finetune_loaders(target_subject, emg_mean, emg_std, imu_mean, imu_std):
    target_emg, target_imu, target_label, target_rep = load_subject_multimodal_with_rep(target_subject)

    ft_train_idx = np.isin(target_rep, FINETUNE_TRAIN_REPS)
    ft_test_idx = np.isin(target_rep, FINETUNE_TEST_REPS)

    target_emg_ft_train = target_emg[ft_train_idx]
    target_imu_ft_train = target_imu[ft_train_idx]
    target_label_ft_train = target_label[ft_train_idx]
    target_rep_ft_train = target_rep[ft_train_idx]

    target_emg_ft_test = target_emg[ft_test_idx]
    target_imu_ft_test = target_imu[ft_test_idx]
    target_label_ft_test = target_label[ft_test_idx]
    target_rep_ft_test = target_rep[ft_test_idx]

    norm_emg_ft_train = (target_emg_ft_train - emg_mean) / emg_std
    norm_imu_ft_train = (target_imu_ft_train - imu_mean) / imu_std
    X_emg_ft_train, X_imu_ft_train, y_ft_train = sliding_window_multimodal_by_segment(
        norm_emg_ft_train,
        norm_imu_ft_train,
        target_label_ft_train,
        target_rep_ft_train
    )

    norm_emg_ft_test = (target_emg_ft_test - emg_mean) / emg_std
    norm_imu_ft_test = (target_imu_ft_test - imu_mean) / imu_std
    X_emg_ft_test, X_imu_ft_test, y_ft_test = sliding_window_multimodal_by_segment(
        norm_emg_ft_test,
        norm_imu_ft_test,
        target_label_ft_test,
        target_rep_ft_test
    )

    ft_train_loader = make_loader(
        MultiModalDataset(X_emg_ft_train, X_imu_ft_train, y_ft_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    ft_test_loader = make_loader(
        MultiModalDataset(X_emg_ft_test, X_imu_ft_test, y_ft_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    train_count = len(y_ft_train)
    test_count = len(y_ft_test)

    del target_emg, target_imu, target_label, target_rep
    del X_emg_ft_train, X_imu_ft_train, y_ft_train
    del X_emg_ft_test, X_imu_ft_test, y_ft_test

    return ft_train_loader, ft_test_loader, train_count, test_count


def create_fusion_model(emg_ch, imu_ch, device):
    return FeatureFusionNet(
        emg_ch=emg_ch,
        imu_ch=imu_ch,
        num_classes=NUM_CLASSES
    ).to(device)


def pretrain_one_fold(fold_idx, pretrain_subjects, fold_save_dir, device):
    print("\n[1/4] 加载预训练数据并计算归一化参数...")
    pretrain_loader, emg_mean, emg_std, imu_mean, imu_std, emg_ch, imu_ch, sample_count = build_pretrain_loader(
        pretrain_subjects
    )

    print("\n[2/4] 构建预训练模型...")
    model = create_fusion_model(emg_ch, imu_ch, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\n{'='*45}")
    print(f"Fold {fold_idx} Phase 1: 盲预训练启动 (强制跑满 {EPOCHS} Epochs)")
    print(f"预训练 Subjects: {pretrain_subjects}")
    print(f"预训练窗口数: {sample_count}")
    print(f"{'='*45}")

    optimizer_pre = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_pre = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pre, T_max=EPOCHS, eta_min=1e-5)

    pretrain_loss_log = []

    for epoch in range(EPOCHS):
        train_loss = train_fusion_one_epoch(model, pretrain_loader, criterion, optimizer_pre, device)
        scheduler_pre.step()
        pretrain_loss_log.append(train_loss)
        print(f"[Pre-train {epoch+1}/{EPOCHS}] Loss:{train_loss:.4f} | LR:{optimizer_pre.param_groups[0]['lr']:.6f}")

    pretrained_model_path = os.path.join(fold_save_dir, "Fusion_Pretrained_model.pth")
    pretrain_log_path = os.path.join(fold_save_dir, "Fusion_Pretrain_log.npz")
    torch.save(model.state_dict(), pretrained_model_path)
    np.savez(
        pretrain_log_path,
        pretrain_loss=pretrain_loss_log,
        pretrain_subjects=np.array(pretrain_subjects),
        sample_count=sample_count,
        emg_mean=emg_mean,
        emg_std=emg_std,
        imu_mean=imu_mean,
        imu_std=imu_std
    )
    print("-> 预训练结束，特征级融合网络最终权重已保存。")

    del pretrain_loader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pretrained_model_path": pretrained_model_path,
        "pretrain_log_path": pretrain_log_path,
        "emg_mean": emg_mean,
        "emg_std": emg_std,
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "emg_ch": emg_ch,
        "imu_ch": imu_ch,
        "sample_count": sample_count,
        "pretrain_subjects": pretrain_subjects,
    }


def finetune_one_subject(fold_idx, target_subject, pretrained_info, subject_save_dir, device):
    print("\n[3/4] 构建微调 DataLoader 与模型...")
    ft_train_loader, ft_test_loader, train_count, test_count = build_finetune_loaders(
        target_subject,
        pretrained_info["emg_mean"],
        pretrained_info["emg_std"],
        pretrained_info["imu_mean"],
        pretrained_info["imu_std"]
    )

    model = create_fusion_model(pretrained_info["emg_ch"], pretrained_info["imu_ch"], device)
    model.load_state_dict(torch.load(pretrained_info["pretrained_model_path"], map_location=device))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\n{'='*45}")
    print(f"Fold {fold_idx} Phase 2: 微调启动 (最大 Epochs: {EPOCHS})")
    print(f"训练: Sub {target_subject} Rep {FINETUNE_TRAIN_REPS} | 测试: Sub {target_subject} Rep {FINETUNE_TEST_REPS}")
    print(f"微调窗口数: {train_count} | 测试窗口数: {test_count}")
    print(f"{'='*45}")

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS, eta_min=1e-6)
    early_stopping_ft = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode='max')
    
    ft_loss_log, ft_test_acc_log = [], []

    for epoch in range(EPOCHS):
        train_loss = train_fusion_one_epoch(model, ft_train_loader, criterion, optimizer_ft, device)
        test_loss, test_acc = evaluate_fusion(model, ft_test_loader, criterion, device)
        scheduler_ft.step()
        
        ft_loss_log.append(train_loss)
        ft_test_acc_log.append(test_acc)

        print(f"[Fine-tune {epoch+1}/{EPOCHS}] Loss:{train_loss:.4f} | Test Acc:{test_acc:.4f} | LR:{optimizer_ft.param_groups[0]['lr']:.6f}")

        if early_stopping_ft(test_acc, model, epoch):
            print(f"微调早停触发! 最佳Epoch: {early_stopping_ft.best_epoch + 1}")
            break

    early_stopping_ft.load_best_model(model)
    finetuned_model_path = os.path.join(subject_save_dir, "Fusion_Finetuned_model.pth")
    finetune_log_path = os.path.join(subject_save_dir, "Fusion_Finetune_log.npz")

    torch.save(model.state_dict(), finetuned_model_path)
    np.savez(
        finetune_log_path,
        ft_loss=ft_loss_log,
        ft_test_acc=ft_test_acc_log,
        train_reps=np.array(FINETUNE_TRAIN_REPS),
        test_reps=np.array(FINETUNE_TEST_REPS),
        train_count=train_count,
        test_count=test_count,
        best_acc=early_stopping_ft.best_score,
        best_epoch=early_stopping_ft.best_epoch + 1
    )

    result = {
        "fold": fold_idx,
        "target_subject": target_subject,
        "train_reps": "-".join(map(str, FINETUNE_TRAIN_REPS)),
        "test_reps": "-".join(map(str, FINETUNE_TEST_REPS)),
        "pretrain_subjects": "-".join(map(str, pretrained_info["pretrain_subjects"])),
        "pretrain_windows": pretrained_info["sample_count"],
        "finetune_train_windows": train_count,
        "finetune_test_windows": test_count,
        "best_acc": early_stopping_ft.best_score,
        "best_epoch": early_stopping_ft.best_epoch + 1,
        "pretrained_model_path": pretrained_info["pretrained_model_path"],
        "pretrain_log_path": pretrained_info["pretrain_log_path"],
        "finetuned_model_path": finetuned_model_path,
        "finetune_log_path": finetune_log_path,
    }

    print(f"\nSub {target_subject} 微调完成，最佳 Test Accuracy: {early_stopping_ft.best_score:.4f}")

    del ft_train_loader, ft_test_loader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_summary_csv(rows):
    fieldnames = [
        "fold",
        "target_subject",
        "train_reps",
        "test_reps",
        "pretrain_subjects",
        "pretrain_windows",
        "finetune_train_windows",
        "finetune_test_windows",
        "best_acc",
        "best_epoch",
        "pretrained_model_path",
        "pretrain_log_path",
        "finetuned_model_path",
        "finetune_log_path",
    ]

    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    seed_everything(RANDOM_SEED, deterministic=DETERMINISTIC)
    device = torch.device(DEVICE)
    os.makedirs(RESULT_ROOT, exist_ok=True)

    print(f"=== 多模态特征级融合+MLP 五折预训练+独立微调 启动 ===")
    print(f"随机种子: {RANDOM_SEED} | 确定性模式: {DETERMINISTIC}")
    print(f"全部 Subjects: {ALL_SUBJECTS}")
    print(f"五折目标 Subjects: {FOLD_TARGET_SUBJECTS}")
    print(f"pth/npz 保存目录: {RESULT_ROOT}")
    print(f"summary CSV: {SUMMARY_CSV_PATH}")

    summary_rows = []

    for fold_idx, target_subjects in enumerate(FOLD_TARGET_SUBJECTS, start=1):
        pretrain_subjects = [s for s in ALL_SUBJECTS if s not in target_subjects]
        fold_save_dir = os.path.join(RESULT_ROOT, f"fold{fold_idx}")
        os.makedirs(fold_save_dir, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"Fold {fold_idx}/5")
        print(f"目标微调 Subjects: {target_subjects}")
        print(f"预训练 Subjects: {pretrain_subjects}")
        print(f"{'#'*60}")

        pretrained_info = pretrain_one_fold(fold_idx, pretrain_subjects, fold_save_dir, device)

        for target_subject in target_subjects:
            subject_save_dir = os.path.join(fold_save_dir, f"sub{target_subject}")
            os.makedirs(subject_save_dir, exist_ok=True)
            result = finetune_one_subject(fold_idx, target_subject, pretrained_info, subject_save_dir, device)
            summary_rows.append(result)
            write_summary_csv(summary_rows)

    mean_acc = np.mean([row["best_acc"] for row in summary_rows])
    print(f"\n融合网络五折流程全部完成！20 个独立微调结果平均 Test Accuracy: {mean_acc:.4f}")
    print(f"汇总 CSV 已保存: {SUMMARY_CSV_PATH}")

if __name__ == "__main__":
    main()
