import os
import random
import csv

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from data import IMUDataset
from models import EMGHandNet1D
from train import EarlyStopping, evaluate, train_one_epoch
from utils import (
    list_available_200hz_users,
    load_epn_user_imu_with_rep,
    sliding_window_imu_by_segment,
)

ALL_USERS = EPN_200HZ_USERS[:20]
FOLD_TARGET_USERS = [
    ALL_USERS[0:4],
    ALL_USERS[4:8],
    ALL_USERS[8:12],
    ALL_USERS[12:16],
    ALL_USERS[16:20],
]
RESULT_ROOT = os.path.join(SAVE_ROOT, "EPN_5fold_IMU_Finetune")
SUMMARY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "summary.csv",
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
            print(f"[Seed] Could not enable full deterministic algorithms: {exc}")


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


def class_count(y):
    return np.bincount(y.astype(np.int64), minlength=NUM_CLASSES)


def should_drop_last_train_batch(num_samples, batch_size):
    if num_samples < 2:
        raise ValueError(f"Training set is too small for BatchNorm: {num_samples} sample(s)")
    return num_samples > batch_size and num_samples % batch_size == 1


def validate_split():
    shared_target_sections = set(TARGET_TRAIN_SECTIONS) & set(TARGET_TEST_SECTIONS)
    shared_reps = set(TRAIN_REPS) & set(TEST_REPS)
    if shared_target_sections and shared_reps:
        raise ValueError(
            "Target train/test sections overlap, so TRAIN_REPS and TEST_REPS must be disjoint. "
            f"Overlapping sections: {sorted(shared_target_sections)}; overlapping reps: {sorted(shared_reps)}"
        )


def validate_users():
    available_users = list_available_200hz_users(DATA_ROOT, OUTER_TRAIN_GROUP)
    missing = sorted(set(ALL_USERS) - set(available_users))
    if missing:
        raise RuntimeError(f"Configured 200Hz users missing or invalid: {missing}")
    return available_users


def build_pretrain_loader(pretrain_users):
    print("\n[1/4] Loading pretrain users and computing global normalization...")
    pre_imu_list, pre_label_list, pre_rep_list = [], [], []
    for user in pretrain_users:
        imu, label, rep = load_epn_user_imu_with_rep(
            user,
            sections=PRETRAIN_SECTIONS,
            outer_group=OUTER_TRAIN_GROUP,
        )
        pre_imu_list.append(imu)
        pre_label_list.append(label)
        pre_rep_list.append(rep)

    raw_pre_imu = np.vstack(pre_imu_list)
    imu_mean = raw_pre_imu.mean(axis=0)
    imu_std = raw_pre_imu.std(axis=0) + 1e-8
    imu_ch = raw_pre_imu.shape[1]
    del raw_pre_imu

    print("\n[2/4] Normalizing and slicing pretrain windows...")
    X_pre_list, y_pre_list = [], []
    for imu, label, rep in zip(pre_imu_list, pre_label_list, pre_rep_list):
        norm_imu = (imu - imu_mean) / imu_std
        Xw, Yw = sliding_window_imu_by_segment(
            norm_imu,
            label,
            rep,
            step_size=STEP_SIZE,
        )
        X_pre_list.append(Xw)
        y_pre_list.append(Yw)

    X_pre = np.concatenate(X_pre_list, axis=0)
    y_pre = np.concatenate(y_pre_list, axis=0)
    del pre_imu_list, pre_label_list, pre_rep_list
    del X_pre_list, y_pre_list

    print(f"Pretrain windows: {len(y_pre)} | class count: {class_count(y_pre).tolist()}")

    sample_count = len(y_pre)
    pretrain_class_count = class_count(y_pre)
    pretrain_loader = make_loader(
        IMUDataset(X_pre, y_pre),
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=should_drop_last_train_batch(sample_count, PRETRAIN_BATCH_SIZE),
    )

    del X_pre, y_pre

    return (
        pretrain_loader,
        imu_mean,
        imu_std,
        imu_ch,
        sample_count,
        pretrain_class_count,
    )


def build_finetune_loaders(target_user, imu_mean, imu_std):
    print(f"\nLoading target user {target_user}...")
    target_train_imu, target_train_label, target_train_rep = (
        load_epn_user_imu_with_rep(
            target_user,
            sections=TARGET_TRAIN_SECTIONS,
            outer_group=OUTER_TRAIN_GROUP,
        )
    )
    target_test_imu, target_test_label, target_test_rep = (
        load_epn_user_imu_with_rep(
            target_user,
            sections=TARGET_TEST_SECTIONS,
            outer_group=OUTER_TRAIN_GROUP,
        )
    )

    train_idx = np.isin(target_train_rep, TRAIN_REPS)
    test_idx = np.isin(target_test_rep, TEST_REPS)

    norm_train_imu = (target_train_imu[train_idx] - imu_mean) / imu_std
    norm_test_imu = (target_test_imu[test_idx] - imu_mean) / imu_std

    X_ft_train, y_ft_train = sliding_window_imu_by_segment(
        norm_train_imu,
        target_train_label[train_idx],
        target_train_rep[train_idx],
        step_size=STEP_SIZE,
    )
    X_ft_test, y_ft_test = sliding_window_imu_by_segment(
        norm_test_imu,
        target_test_label[test_idx],
        target_test_rep[test_idx],
        step_size=STEP_SIZE,
    )

    print(
        f"Fine-tune windows: {len(y_ft_train)} | class count: {class_count(y_ft_train).tolist()}"
    )
    print(f"Test windows: {len(y_ft_test)} | class count: {class_count(y_ft_test).tolist()}")

    train_count = len(y_ft_train)
    test_count = len(y_ft_test)
    train_class_count = class_count(y_ft_train)
    test_class_count = class_count(y_ft_test)
    ft_train_loader = make_loader(
        IMUDataset(X_ft_train, y_ft_train),
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        drop_last=should_drop_last_train_batch(train_count, FINETUNE_BATCH_SIZE),
    )
    ft_test_loader = make_loader(
        IMUDataset(X_ft_test, y_ft_test),
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    del target_train_imu, target_train_label, target_train_rep
    del target_test_imu, target_test_label, target_test_rep
    del X_ft_train, y_ft_train
    del X_ft_test, y_ft_test

    return ft_train_loader, ft_test_loader, train_count, test_count, train_class_count, test_class_count


def create_imu_model(imu_ch, device):
    return EMGHandNet1D(
        in_ch=imu_ch,
        num_classes=NUM_CLASSES,
    ).to(device)


def pretrain_one_fold(fold_idx, pretrain_users, fold_save_dir, device):
    (
        pretrain_loader,
        imu_mean,
        imu_std,
        imu_ch,
        sample_count,
        pretrain_class_count,
    ) = build_pretrain_loader(pretrain_users)

    print("\n[3/4] Building IMU model...")
    model = create_imu_model(imu_ch, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\nFold {fold_idx} Phase 1: cross-user pretraining ({EPOCHS} epochs)")
    print(f"Pretrain users ({len(pretrain_users)}): {pretrain_users}")
    optimizer_pre = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_pre = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pre,
        T_max=EPOCHS,
        eta_min=1e-5,
    )
    pretrain_loss_log = []
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            pretrain_loader,
            criterion,
            optimizer_pre,
            device,
        )
        scheduler_pre.step()
        pretrain_loss_log.append(train_loss)
        print(
            f"[Pre-train {epoch + 1}/{EPOCHS}] "
            f"Loss:{train_loss:.4f} | LR:{optimizer_pre.param_groups[0]['lr']:.6f}"
        )

    pretrained_model_path = os.path.join(fold_save_dir, "IMU_Pretrained_model.pth")
    pretrain_log_path = os.path.join(fold_save_dir, "IMU_Pretrain_log.npz")
    torch.save(model.state_dict(), pretrained_model_path)
    np.savez(
        pretrain_log_path,
        pretrain_loss=np.asarray(pretrain_loss_log),
        imu_mean=imu_mean,
        imu_std=imu_std,
        pretrain_users=np.asarray(pretrain_users),
        pretrain_windows=sample_count,
        pretrain_class_count=pretrain_class_count,
    )
    print("Pretraining finished and model saved.")

    del pretrain_loader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pretrained_model_path": pretrained_model_path,
        "pretrain_log_path": pretrain_log_path,
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "imu_ch": imu_ch,
        "pretrain_users": pretrain_users,
        "pretrain_windows": sample_count,
        "pretrain_class_count": pretrain_class_count,
    }


def finetune_one_user(fold_idx, target_user, pretrained_info, user_save_dir, device):
    ft_train_loader, ft_test_loader, train_count, test_count, train_class_count, test_class_count = build_finetune_loaders(
        target_user,
        pretrained_info["imu_mean"],
        pretrained_info["imu_std"],
    )

    model = create_imu_model(pretrained_info["imu_ch"], device)
    model.load_state_dict(torch.load(pretrained_info["pretrained_model_path"], map_location=device))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\n[4/4] Fold {fold_idx} Phase 2: target-user fine-tuning ({target_user})")
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft,
        T_max=EPOCHS,
        eta_min=1e-6,
    )
    early_stopping_ft = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")
    ft_loss_log, ft_test_acc_log = [], []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            ft_train_loader,
            criterion,
            optimizer_ft,
            device,
        )
        test_loss, test_acc = evaluate(model, ft_test_loader, criterion, device)
        scheduler_ft.step()

        ft_loss_log.append(train_loss)
        ft_test_acc_log.append(test_acc)
        print(
            f"[Fine-tune {epoch + 1}/{EPOCHS}] "
            f"Loss:{train_loss:.4f} | Test Loss:{test_loss:.4f} | "
            f"Test Acc:{test_acc:.4f} | LR:{optimizer_ft.param_groups[0]['lr']:.6f}"
        )

        if early_stopping_ft(test_acc, model, epoch):
            print(f"Fine-tuning early stopped at epoch {epoch + 1}.")
            break

    early_stopping_ft.load_best_model(model)
    finetuned_model_path = os.path.join(user_save_dir, "IMU_Finetuned_model.pth")
    finetune_log_path = os.path.join(user_save_dir, "IMU_Finetune_log.npz")
    torch.save(model.state_dict(), finetuned_model_path)
    np.savez(
        finetune_log_path,
        ft_loss=np.asarray(ft_loss_log),
        ft_test_acc=np.asarray(ft_test_acc_log),
        target_user=target_user,
        train_reps=np.asarray(TRAIN_REPS),
        test_reps=np.asarray(TEST_REPS),
        train_windows=train_count,
        test_windows=test_count,
        train_class_count=train_class_count,
        test_class_count=test_class_count,
        best_acc=early_stopping_ft.best_score,
        best_epoch=early_stopping_ft.best_epoch + 1,
    )

    result = {
        "fold": fold_idx,
        "target_user": target_user,
        "train_reps": "-".join(map(str, TRAIN_REPS)),
        "test_reps": "-".join(map(str, TEST_REPS)),
        "pretrain_users": "-".join(pretrained_info["pretrain_users"]),
        "pretrain_windows": pretrained_info["pretrain_windows"],
        "finetune_train_windows": train_count,
        "finetune_test_windows": test_count,
        "best_acc": early_stopping_ft.best_score,
        "best_epoch": early_stopping_ft.best_epoch + 1,
        "pretrained_model_path": pretrained_info["pretrained_model_path"],
        "pretrain_log_path": pretrained_info["pretrain_log_path"],
        "finetuned_model_path": finetuned_model_path,
        "finetune_log_path": finetune_log_path,
    }

    print(f"\nEPN IMU baseline completed for {target_user}.")
    print(f"Best target test accuracy: {early_stopping_ft.best_score:.4f}")
    print(f"Best epoch: {early_stopping_ft.best_epoch + 1}")
    print(f"Saved to: {user_save_dir}")

    del ft_train_loader, ft_test_loader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_summary_csv(rows):
    fieldnames = [
        "fold",
        "target_user",
        "train_reps",
        "test_reps",
        "pretrain_users",
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
    os.makedirs(RESULT_ROOT, exist_ok=True)
    validate_split()
    available_users = validate_users()

    device = torch.device(DEVICE)
    print("=== EPN-107 200Hz IMU baseline ===")
    print(f"Device: {DEVICE}")
    print(f"Seed: {RANDOM_SEED} | Deterministic: {DETERMINISTIC}")
    print(
        f"Window: {WINDOW_SIZE} | Step: {STEP_SIZE} | "
        f"Pretrain batch: {PRETRAIN_BATCH_SIZE} | Fine-tune batch: {FINETUNE_BATCH_SIZE}"
    )
    print(f"Available 200Hz comparison users ({len(available_users)}): {available_users}")
    print(f"Using users ({len(ALL_USERS)}): {ALL_USERS}")
    print(f"Fold target users: {FOLD_TARGET_USERS}")
    print(f"pth/npz save root: {RESULT_ROOT}")
    print(f"summary CSV: {SUMMARY_CSV_PATH}")

    summary_rows = []

    for fold_idx, target_users in enumerate(FOLD_TARGET_USERS, start=1):
        if not target_users:
            continue
        pretrain_users = [u for u in ALL_USERS if u not in target_users]
        fold_save_dir = os.path.join(RESULT_ROOT, f"fold{fold_idx}")
        os.makedirs(fold_save_dir, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"Fold {fold_idx}/{len(FOLD_TARGET_USERS)}")
        print(f"Target users: {target_users}")
        print(f"Pretrain users ({len(pretrain_users)}): {pretrain_users}")
        print(f"{'#' * 60}")

        pretrained_info = pretrain_one_fold(fold_idx, pretrain_users, fold_save_dir, device)

        for target_user in target_users:
            user_save_dir = os.path.join(fold_save_dir, target_user)
            os.makedirs(user_save_dir, exist_ok=True)
            result = finetune_one_user(fold_idx, target_user, pretrained_info, user_save_dir, device)
            summary_rows.append(result)
            write_summary_csv(summary_rows)

    mean_acc = np.mean([row["best_acc"] for row in summary_rows])
    print("\nEPN 5-fold IMU baseline completed.")
    print(f"Mean target test accuracy across {len(summary_rows)} users: {mean_acc:.4f}")
    print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
