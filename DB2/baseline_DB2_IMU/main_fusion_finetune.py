# IMU 单模态两阶段训练：DB2 1-20 号受试者五折预训练 + 单受试者独立微调
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
from data import IMUDataset
from models import EMGHandNet1D
from train import EarlyStopping, evaluate, train_one_epoch


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

RESULT_ROOT = "/hdd/zhaozixuan/baseline_DB2_IMU/result/DB2_5fold_IMU_Finetune"
SUMMARY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DB2_5fold_IMU_Finetune_summary.csv",
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


def sliding_window_imu(X_imu, y):
    X_imu_w, y_w = [], []
    total = (len(X_imu) - WINDOW_SIZE) // STEP_SIZE + 1
    for i in range(total):
        s = i * STEP_SIZE
        e = s + WINDOW_SIZE
        wy = y[s:e]
        if DROP_MIXED_WINDOW and not np.all(wy == wy[0]):
            continue
        label = wy[0] if DROP_MIXED_WINDOW else np.bincount(wy).argmax()
        X_imu_w.append(X_imu[s:e])
        y_w.append(label)
    return np.array(X_imu_w), np.array(y_w)


def sliding_window_imu_by_segment(X_imu, y, rep):
    X_imu_w, y_w = [], []
    if len(y) < WINDOW_SIZE:
        return (
            np.empty((0, WINDOW_SIZE, X_imu.shape[1]), dtype=X_imu.dtype),
            np.empty((0,), dtype=y.dtype),
        )

    change_points = np.where((y[1:] != y[:-1]) | (rep[1:] != rep[:-1]))[0] + 1
    boundaries = np.concatenate(([0], change_points, [len(y)]))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < WINDOW_SIZE:
            continue
        Xw_imu, Yw = sliding_window_imu(X_imu[start:end], y[start:end])
        if len(Yw) == 0:
            continue
        X_imu_w.append(Xw_imu)
        y_w.append(Yw)

    if not y_w:
        return (
            np.empty((0, WINDOW_SIZE, X_imu.shape[1]), dtype=X_imu.dtype),
            np.empty((0,), dtype=y.dtype),
        )

    return np.concatenate(X_imu_w, axis=0), np.concatenate(y_w, axis=0)


def load_subject_imu_with_rep(subject_id):
    data_dir = f"{LOSO_BASE_DATA_DIR}/Subject_{subject_id}"
    file_list = [f"S{subject_id}_E{exercise}_A1.mat" for exercise in EXERCISES]

    motion_list, label_list, rep_list = [], [], []

    for file in file_list:
        data = scio.loadmat(os.path.join(data_dir, file))
        motion_list.append(data[MAT_MOTION_KEY])
        label_list.append(data[MAT_LABEL_KEY])
        rep_list.append(data[MAT_REP_KEY])

    imu = np.vstack(motion_list)
    label = np.vstack(label_list).squeeze()
    rep = np.vstack(rep_list).squeeze()

    idx = np.where((label != 0) & (rep != 0))[0]
    return imu[idx], label[idx] - 1, rep[idx]


def build_pretrain_loader(pretrain_subjects):
    pretrain_imu_list, pretrain_label_list, pretrain_rep_list = [], [], []

    for sub_id in pretrain_subjects:
        imu, label, rep = load_subject_imu_with_rep(sub_id)
        pretrain_imu_list.append(imu)
        pretrain_label_list.append(label)
        pretrain_rep_list.append(rep)

    global_pretrain_imu = np.vstack(pretrain_imu_list)
    imu_mean = global_pretrain_imu.mean(axis=0)
    imu_std = global_pretrain_imu.std(axis=0) + 1e-8
    imu_ch = global_pretrain_imu.shape[1]
    del global_pretrain_imu

    X_imu_pretrain_w, y_pretrain_w = [], []
    for imu, label, rep in zip(pretrain_imu_list, pretrain_label_list, pretrain_rep_list):
        norm_imu = (imu - imu_mean) / imu_std
        Xw_imu, yw = sliding_window_imu_by_segment(norm_imu, label, rep)
        X_imu_pretrain_w.append(Xw_imu)
        y_pretrain_w.append(yw)

    X_imu_pretrain = np.concatenate(X_imu_pretrain_w, axis=0)
    y_pretrain = np.concatenate(y_pretrain_w, axis=0)
    del pretrain_imu_list, pretrain_label_list, pretrain_rep_list
    del X_imu_pretrain_w, y_pretrain_w

    pretrain_loader = make_loader(
        IMUDataset(X_imu_pretrain, y_pretrain),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    sample_count = len(y_pretrain)

    del X_imu_pretrain, y_pretrain

    return pretrain_loader, imu_mean, imu_std, imu_ch, sample_count


def build_finetune_loaders(target_subject, imu_mean, imu_std):
    target_imu, target_label, target_rep = load_subject_imu_with_rep(target_subject)

    ft_train_idx = np.isin(target_rep, FINETUNE_TRAIN_REPS)
    ft_test_idx = np.isin(target_rep, FINETUNE_TEST_REPS)

    target_imu_ft_train = target_imu[ft_train_idx]
    target_label_ft_train = target_label[ft_train_idx]
    target_rep_ft_train = target_rep[ft_train_idx]

    target_imu_ft_test = target_imu[ft_test_idx]
    target_label_ft_test = target_label[ft_test_idx]
    target_rep_ft_test = target_rep[ft_test_idx]

    norm_imu_ft_train = (target_imu_ft_train - imu_mean) / imu_std
    X_imu_ft_train, y_ft_train = sliding_window_imu_by_segment(
        norm_imu_ft_train,
        target_label_ft_train,
        target_rep_ft_train,
    )

    norm_imu_ft_test = (target_imu_ft_test - imu_mean) / imu_std
    X_imu_ft_test, y_ft_test = sliding_window_imu_by_segment(
        norm_imu_ft_test,
        target_label_ft_test,
        target_rep_ft_test,
    )

    ft_train_loader = make_loader(
        IMUDataset(X_imu_ft_train, y_ft_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    ft_test_loader = make_loader(
        IMUDataset(X_imu_ft_test, y_ft_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    train_count = len(y_ft_train)
    test_count = len(y_ft_test)

    del target_imu, target_label, target_rep
    del X_imu_ft_train, y_ft_train
    del X_imu_ft_test, y_ft_test

    return ft_train_loader, ft_test_loader, train_count, test_count


def create_imu_model(imu_ch, device):
    return EMGHandNet1D(
        in_ch=imu_ch,
        num_classes=NUM_CLASSES,
    ).to(device)


def pretrain_one_fold(fold_idx, pretrain_subjects, fold_save_dir, device):
    print("\n[1/4] 加载 IMU 预训练数据并计算归一化参数...")
    pretrain_loader, imu_mean, imu_std, imu_ch, sample_count = build_pretrain_loader(
        pretrain_subjects
    )

    print("\n[2/4] 构建 IMU 单模态预训练模型...")
    model = create_imu_model(imu_ch, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\n{'='*45}")
    print(f"Fold {fold_idx} Phase 1: IMU 盲预训练启动 (强制跑满 {EPOCHS} Epochs)")
    print(f"预训练 Subjects: {pretrain_subjects}")
    print(f"预训练窗口数: {sample_count}")
    print(f"{'='*45}")

    optimizer_pre = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_pre = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pre,
        T_max=EPOCHS,
        eta_min=1e-5,
    )

    pretrain_loss_log = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, pretrain_loader, criterion, optimizer_pre, device)
        scheduler_pre.step()
        pretrain_loss_log.append(train_loss)
        print(
            f"[Pre-train {epoch+1}/{EPOCHS}] "
            f"Loss:{train_loss:.4f} | LR:{optimizer_pre.param_groups[0]['lr']:.6f}"
        )

    pretrained_model_path = os.path.join(fold_save_dir, "IMU_Pretrained_model.pth")
    pretrain_log_path = os.path.join(fold_save_dir, "IMU_Pretrain_log.npz")
    torch.save(model.state_dict(), pretrained_model_path)
    np.savez(
        pretrain_log_path,
        pretrain_loss=pretrain_loss_log,
        pretrain_subjects=np.array(pretrain_subjects),
        sample_count=sample_count,
        imu_mean=imu_mean,
        imu_std=imu_std,
    )
    print("-> IMU 单模态预训练结束，最终权重已保存。")

    del pretrain_loader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pretrained_model_path": pretrained_model_path,
        "pretrain_log_path": pretrain_log_path,
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "imu_ch": imu_ch,
        "sample_count": sample_count,
        "pretrain_subjects": pretrain_subjects,
    }


def finetune_one_subject(fold_idx, target_subject, pretrained_info, subject_save_dir, device):
    print("\n[3/4] 构建 IMU 微调 DataLoader 与模型...")
    ft_train_loader, ft_test_loader, train_count, test_count = build_finetune_loaders(
        target_subject,
        pretrained_info["imu_mean"],
        pretrained_info["imu_std"],
    )

    model = create_imu_model(pretrained_info["imu_ch"], device)
    model.load_state_dict(torch.load(pretrained_info["pretrained_model_path"], map_location=device))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\n{'='*45}")
    print(f"Fold {fold_idx} Phase 2: IMU 微调启动 (最大 Epochs: {EPOCHS})")
    print(f"训练: Sub {target_subject} Rep {FINETUNE_TRAIN_REPS} | 测试: Sub {target_subject} Rep {FINETUNE_TEST_REPS}")
    print(f"微调窗口数: {train_count} | 测试窗口数: {test_count}")
    print(f"{'='*45}")

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft,
        T_max=EPOCHS,
        eta_min=1e-6,
    )
    early_stopping_ft = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")

    ft_loss_log, ft_test_acc_log = [], []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, ft_train_loader, criterion, optimizer_ft, device)
        test_loss, test_acc = evaluate(model, ft_test_loader, criterion, device)
        scheduler_ft.step()

        ft_loss_log.append(train_loss)
        ft_test_acc_log.append(test_acc)

        print(
            f"[Fine-tune {epoch+1}/{EPOCHS}] "
            f"Loss:{train_loss:.4f} | Test Acc:{test_acc:.4f} | "
            f"LR:{optimizer_ft.param_groups[0]['lr']:.6f}"
        )

        if early_stopping_ft(test_acc, model, epoch):
            print(f"微调早停触发! 最佳Epoch: {early_stopping_ft.best_epoch + 1}")
            break

    early_stopping_ft.load_best_model(model)
    finetuned_model_path = os.path.join(subject_save_dir, "IMU_Finetuned_model.pth")
    finetune_log_path = os.path.join(subject_save_dir, "IMU_Finetune_log.npz")

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
        best_epoch=early_stopping_ft.best_epoch + 1,
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

    print(f"\nSub {target_subject} IMU 微调完成，最佳 Test Accuracy: {early_stopping_ft.best_score:.4f}")

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

    print("=== DB2 IMU 单模态五折预训练+独立微调 启动 ===")
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
    print(f"\nDB2 IMU 单模态五折流程全部完成！20 个独立微调结果平均 Test Accuracy: {mean_acc:.4f}")
    print(f"汇总 CSV 已保存: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
