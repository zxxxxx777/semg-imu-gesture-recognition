import os
import random
import csv

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from config import *
from data import MultiModalDataset
from models import FeatureFusionNet
from train import EarlyStopping, evaluate_fusion, train_fusion_one_epoch
from utils import (
    list_available_200hz_users,
    load_epn_user_multimodal_with_rep,
    sliding_window_multimodal_by_segment,
)

ALL_USERS = EPN_200HZ_USERS[:20]
FOLD_TARGET_USERS = [
    ALL_USERS[0:4],
    ALL_USERS[4:8],
    ALL_USERS[8:12],
    ALL_USERS[12:16],
    ALL_USERS[16:20],
]
RESULT_ROOT = os.path.join(SAVE_ROOT, "EPN_5fold_Fusion_BiResCrossAttn")
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


def load_pretrain_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    print(f"\n[1/4] 加载预训练用户 ({len(pretrain_users)} users): {pretrain_users}")
    pre_emg_list, pre_imu_list, pre_label_list, pre_rep_list = [], [], [], []
    for user in pretrain_users:
        emg, imu, label, rep = load_epn_user_multimodal_with_rep(
            user,
            sections=PRETRAIN_SECTIONS,
            outer_group=OUTER_TRAIN_GROUP,
        )
        pre_emg_list.append(emg)
        pre_imu_list.append(imu)
        pre_label_list.append(label)
        pre_rep_list.append(rep)

    print("计算预训练用户全局 Z-Score 参数...")
    raw_pre_emg = np.vstack(pre_emg_list)
    raw_pre_imu = np.vstack(pre_imu_list)
    emg_mean = raw_pre_emg.mean(axis=0)
    emg_std = raw_pre_emg.std(axis=0) + 1e-8
    imu_mean = raw_pre_imu.mean(axis=0)
    imu_std = raw_pre_imu.std(axis=0) + 1e-8
    emg_ch = raw_pre_emg.shape[1]
    imu_ch = raw_pre_imu.shape[1]
    del raw_pre_emg, raw_pre_imu

    print("\n[2/4] 归一化并切分预训练窗口...")
    X_pre_emg_list, X_pre_imu_list, y_pre_list = [], [], []
    for emg, imu, label, rep in zip(pre_emg_list, pre_imu_list, pre_label_list, pre_rep_list):
        norm_emg = (emg - emg_mean) / emg_std
        norm_imu = (imu - imu_mean) / imu_std
        Xw_emg, Xw_imu, Yw = sliding_window_multimodal_by_segment(
            norm_emg,
            norm_imu,
            label,
            rep,
            step_size=PRETRAIN_STEP_SIZE,
        )
        X_pre_emg_list.append(Xw_emg)
        X_pre_imu_list.append(Xw_imu)
        y_pre_list.append(Yw)

    X_pre_emg = np.concatenate(X_pre_emg_list, axis=0)
    X_pre_imu = np.concatenate(X_pre_imu_list, axis=0)
    y_pre = np.concatenate(y_pre_list, axis=0)
    del pre_emg_list, pre_imu_list, pre_label_list, pre_rep_list
    del X_pre_emg_list, X_pre_imu_list, y_pre_list

    sample_count = len(y_pre)
    pretrain_class_count = class_count(y_pre)
    pretrain_loader = make_loader(
        MultiModalDataset(X_pre_emg, X_pre_imu, y_pre),
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=should_drop_last_train_batch(sample_count, PRETRAIN_BATCH_SIZE),
    )
    print(f"预训练集构建完成: {sample_count} 窗口 | class count: {pretrain_class_count.tolist()}")

    del X_pre_emg, X_pre_imu, y_pre

    return (
        pretrain_loader,
        emg_mean,
        emg_std,
        imu_mean,
        imu_std,
        emg_ch,
        imu_ch,
        sample_count,
        pretrain_class_count,
    )


def build_finetune_loaders(target_user, emg_mean, emg_std, imu_mean, imu_std):
    print(f"\n加载目标用户 {target_user}...")
    target_train_emg, target_train_imu, target_train_label, target_train_rep = (
        load_epn_user_multimodal_with_rep(
            target_user,
            sections=TARGET_TRAIN_SECTIONS,
            outer_group=OUTER_TRAIN_GROUP,
        )
    )
    target_test_emg, target_test_imu, target_test_label, target_test_rep = (
        load_epn_user_multimodal_with_rep(
            target_user,
            sections=TARGET_TEST_SECTIONS,
            outer_group=OUTER_TRAIN_GROUP,
        )
    )

    train_idx = np.isin(target_train_rep, TRAIN_REPS)
    test_idx = np.isin(target_test_rep, TEST_REPS)

    X_ft_train_emg, X_ft_train_imu, y_ft_train = sliding_window_multimodal_by_segment(
        (target_train_emg[train_idx] - emg_mean) / emg_std,
        (target_train_imu[train_idx] - imu_mean) / imu_std,
        target_train_label[train_idx],
        target_train_rep[train_idx],
        step_size=FINETUNE_STEP_SIZE,
    )
    X_ft_test_emg, X_ft_test_imu, y_ft_test = sliding_window_multimodal_by_segment(
        (target_test_emg[test_idx] - emg_mean) / emg_std,
        (target_test_imu[test_idx] - imu_mean) / imu_std,
        target_test_label[test_idx],
        target_test_rep[test_idx],
        step_size=FINETUNE_STEP_SIZE,
    )
    print(f"微调训练集: {len(y_ft_train)} 窗口 | class count: {class_count(y_ft_train).tolist()}")
    print(f"测试集: {len(y_ft_test)} 窗口 | class count: {class_count(y_ft_test).tolist()}")

    train_count = len(y_ft_train)
    test_count = len(y_ft_test)
    train_class_count = class_count(y_ft_train)
    test_class_count = class_count(y_ft_test)
    ft_train_loader = make_loader(
        MultiModalDataset(X_ft_train_emg, X_ft_train_imu, y_ft_train),
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        drop_last=should_drop_last_train_batch(train_count, FINETUNE_BATCH_SIZE),
    )
    ft_test_loader = make_loader(
        MultiModalDataset(X_ft_test_emg, X_ft_test_imu, y_ft_test),
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    del target_train_emg, target_train_imu, target_train_label, target_train_rep
    del target_test_emg, target_test_imu, target_test_label, target_test_rep
    del X_ft_train_emg, X_ft_train_imu, y_ft_train
    del X_ft_test_emg, X_ft_test_imu, y_ft_test

    return ft_train_loader, ft_test_loader, train_count, test_count, train_class_count, test_class_count


def create_fusion_model(emg_ch, imu_ch, device):
    return FeatureFusionNet(
        emg_ch=emg_ch,
        imu_ch=imu_ch,
        num_classes=NUM_CLASSES,
        lstm_hidden=LSTM_HIDDEN,
    ).to(device)


def pretrain_one_fold(fold_idx, pretrain_users, fold_save_dir, device):
    pretrain_cache_path = os.path.join(fold_save_dir, PRETRAIN_CACHE_NAME)
    pretrain_log_path = os.path.join(fold_save_dir, "Fusion_Pretrain_log.npz")
    use_cache = USE_PRETRAIN_CACHE and (not FORCE_PRETRAIN) and os.path.exists(pretrain_cache_path)
    pretrain_loss_log = []

    if use_cache:
        print(f"\n[1/4] 读取 Fold {fold_idx} 预训练缓存: {pretrain_cache_path}")
        cache = load_pretrain_cache(pretrain_cache_path)
        emg_mean = cache["emg_mean"]
        emg_std = cache["emg_std"]
        imu_mean = cache["imu_mean"]
        imu_std = cache["imu_std"]
        emg_ch = int(cache["emg_ch"])
        imu_ch = int(cache["imu_ch"])
        sample_count = int(cache.get("sample_count", -1))
        pretrain_class_count = np.asarray(cache.get("pretrain_class_count", []))
        np.savez(
            pretrain_log_path,
            pretrain_loss=np.asarray(pretrain_loss_log),
            emg_mean=emg_mean,
            emg_std=emg_std,
            imu_mean=imu_mean,
            imu_std=imu_std,
            pretrain_users=np.asarray(pretrain_users),
            pretrain_windows=sample_count,
            pretrain_class_count=pretrain_class_count,
            loaded_from_cache=True,
        )
    else:
        (
            pretrain_loader,
            emg_mean,
            emg_std,
            imu_mean,
            imu_std,
            emg_ch,
            imu_ch,
            sample_count,
            pretrain_class_count,
        ) = build_pretrain_loader(pretrain_users)

        print("\n[3/4] 初始化双向残差交叉注意力互学习网络...")
        model = create_fusion_model(emg_ch, imu_ch, device)
        pretrain_criterion = nn.CrossEntropyLoss(label_smoothing=PRETRAIN_LABEL_SMOOTHING)

        print(f"\n[3/4] Fold {fold_idx} 开始 Phase 1: 跨用户预训练 ({PRETRAIN_EPOCHS} Epochs)")
        print(f"预训练用户 ({len(pretrain_users)}): {pretrain_users}")
        optimizer_pre = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
        scheduler_pre = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_pre,
            T_max=PRETRAIN_EPOCHS,
            eta_min=1e-5,
        )

        for epoch in range(PRETRAIN_EPOCHS):
            train_loss = train_fusion_one_epoch(
                model,
                pretrain_loader,
                pretrain_criterion,
                optimizer_pre,
                device,
                alpha=PRETRAIN_ALPHA,
                beta_enh=PRETRAIN_BETA_ENH,
                beta_kl=PRETRAIN_BETA_KL,
                beta_cons=PRETRAIN_BETA_CONS,
                beta_gate=PRETRAIN_BETA_GATE,
                beta_rel_align=PRETRAIN_BETA_REL_ALIGN,
            )
            scheduler_pre.step()
            pretrain_loss_log.append(train_loss)
            print(
                f"[Pre-train {epoch + 1}/{PRETRAIN_EPOCHS}] "
                f"Loss:{train_loss:.4f} | LR:{optimizer_pre.param_groups[0]['lr']:.6f}"
            )

        torch.save(
            {
                "model_state": model.state_dict(),
                "emg_mean": emg_mean,
                "emg_std": emg_std,
                "imu_mean": imu_mean,
                "imu_std": imu_std,
                "emg_ch": emg_ch,
                "imu_ch": imu_ch,
                "pretrain_users": pretrain_users,
                "sample_count": sample_count,
                "pretrain_class_count": pretrain_class_count,
            },
            pretrain_cache_path,
        )
        np.savez(
            pretrain_log_path,
            pretrain_loss=np.asarray(pretrain_loss_log),
            emg_mean=emg_mean,
            emg_std=emg_std,
            imu_mean=imu_mean,
            imu_std=imu_std,
            pretrain_users=np.asarray(pretrain_users),
            pretrain_windows=sample_count,
            pretrain_class_count=pretrain_class_count,
            loaded_from_cache=False,
        )
        print(f"已保存 Fold {fold_idx} 预训练缓存: {pretrain_cache_path}")

        del pretrain_loader, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "pretrained_model_path": pretrain_cache_path,
        "pretrain_log_path": pretrain_log_path,
        "emg_mean": emg_mean,
        "emg_std": emg_std,
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "emg_ch": emg_ch,
        "imu_ch": imu_ch,
        "pretrain_users": pretrain_users,
        "pretrain_windows": sample_count,
        "pretrain_class_count": pretrain_class_count,
    }


def finetune_one_user(fold_idx, target_user, pretrained_info, user_save_dir, device):
    ft_train_loader, ft_test_loader, train_count, test_count, train_class_count, test_class_count = build_finetune_loaders(
        target_user,
        pretrained_info["emg_mean"],
        pretrained_info["emg_std"],
        pretrained_info["imu_mean"],
        pretrained_info["imu_std"],
    )

    print("\n[4/4] 初始化并加载预训练权重，开始目标用户微调...")
    model = create_fusion_model(pretrained_info["emg_ch"], pretrained_info["imu_ch"], device)
    cache = load_pretrain_cache(pretrained_info["pretrained_model_path"])
    model.load_state_dict(cache["model_state"])
    finetune_criterion = nn.CrossEntropyLoss(label_smoothing=FINETUNE_LABEL_SMOOTHING)

    print(f"\n[4/4] Fold {fold_idx} 开始 Phase 2: 目标用户微调 ({target_user})")
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS, eta_min=1e-6)
    early_stopping_ft = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")
    ft_loss_log, ft_test_acc_log, gate_emg_log, gate_imu_log = [], [], [], []
    best_gate_emg = 0.0
    best_gate_imu = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_fusion_one_epoch(
            model,
            ft_train_loader,
            finetune_criterion,
            optimizer_ft,
            device,
            alpha=FINETUNE_ALPHA,
            beta_enh=FINETUNE_BETA_ENH,
            beta_kl=FINETUNE_BETA_KL,
            beta_cons=FINETUNE_BETA_CONS,
            beta_gate=FINETUNE_BETA_GATE,
            beta_rel_align=FINETUNE_BETA_REL_ALIGN,
        )
        _, test_acc, gate_emg, gate_imu = evaluate_fusion(
            model,
            ft_test_loader,
            finetune_criterion,
            device,
        )
        scheduler_ft.step()
        ft_loss_log.append(train_loss)
        ft_test_acc_log.append(test_acc)
        gate_emg_log.append(gate_emg)
        gate_imu_log.append(gate_imu)

        print(
            f"[Fine-tune {epoch + 1}/{EPOCHS}] Loss:{train_loss:.4f} | Test Acc:{test_acc:.4f} | "
            f"LR:{optimizer_ft.param_groups[0]['lr']:.6f} | GateE:{gate_emg:.3f} | GateI:{gate_imu:.3f}"
        )

        should_stop = early_stopping_ft(test_acc, model, epoch)
        if early_stopping_ft.best_epoch == epoch:
            best_gate_emg = gate_emg
            best_gate_imu = gate_imu
        if should_stop:
            print(
                f"微调早停触发! 最佳验证准确率: {early_stopping_ft.best_score:.4f} "
                f"(Epoch {early_stopping_ft.best_epoch + 1})"
            )
            break

    best_model_path = os.path.join(user_save_dir, "best_bires_crossattn_mutual_model.pth")
    finetune_log_path = os.path.join(user_save_dir, "Fusion_Finetune_log.npz")
    torch.save(early_stopping_ft.best_model, best_model_path)
    np.savez(
        finetune_log_path,
        ft_loss=np.asarray(ft_loss_log),
        ft_test_acc=np.asarray(ft_test_acc_log),
        gate_emg=np.asarray(gate_emg_log),
        gate_imu=np.asarray(gate_imu_log),
        target_user=target_user,
        train_reps=np.asarray(TRAIN_REPS),
        test_reps=np.asarray(TEST_REPS),
        train_windows=train_count,
        test_windows=test_count,
        train_class_count=train_class_count,
        test_class_count=test_class_count,
        best_acc=early_stopping_ft.best_score,
        best_epoch=early_stopping_ft.best_epoch + 1,
        best_gate_emg=best_gate_emg,
        best_gate_imu=best_gate_imu,
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
        "best_gate_emg": best_gate_emg,
        "best_gate_imu": best_gate_imu,
        "pretrained_model_path": pretrained_info["pretrained_model_path"],
        "pretrain_log_path": pretrained_info["pretrain_log_path"],
        "finetuned_model_path": best_model_path,
        "finetune_log_path": finetune_log_path,
    }

    print(f"\n训练结束。{target_user} 最佳测试准确率: {early_stopping_ft.best_score:.4f}")
    print(f"最佳 Epoch: {early_stopping_ft.best_epoch + 1}")
    print(f"保存目录: {user_save_dir}")

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
        "best_gate_emg",
        "best_gate_imu",
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
    validate_split()
    seed_everything(RANDOM_SEED, deterministic=DETERMINISTIC)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    device = torch.device(DEVICE)
    available_users = validate_users()

    print("========== EPN 双向残差交叉注意力互学习融合网络 ==========")
    print(f"使用设备: {DEVICE}")
    print(f"随机种子: {RANDOM_SEED} | 确定性模式: {DETERMINISTIC}")
    print(
        f"窗口: {WINDOW_SIZE} | 预训练步长: {PRETRAIN_STEP_SIZE} | 微调步长: {FINETUNE_STEP_SIZE} | "
        f"预训练Batch: {PRETRAIN_BATCH_SIZE} | 微调Batch: {FINETUNE_BATCH_SIZE}"
    )
    print(f"可用 200Hz 多模态用户 ({len(available_users)}): {available_users}")
    print(f"使用用户 ({len(ALL_USERS)}): {ALL_USERS}")
    print(f"五折目标用户: {FOLD_TARGET_USERS}")
    print(f"pth/npz 保存目录: {RESULT_ROOT}")
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
        print(f"目标用户: {target_users}")
        print(f"预训练用户 ({len(pretrain_users)}): {pretrain_users}")
        print(f"{'#' * 60}")

        pretrained_info = pretrain_one_fold(fold_idx, pretrain_users, fold_save_dir, device)

        for target_user in target_users:
            user_save_dir = os.path.join(fold_save_dir, target_user)
            os.makedirs(user_save_dir, exist_ok=True)
            result = finetune_one_user(fold_idx, target_user, pretrained_info, user_save_dir, device)
            summary_rows.append(result)
            write_summary_csv(summary_rows)

    mean_acc = np.mean([row["best_acc"] for row in summary_rows])
    print(f"\nEPN 五折流程完成。{len(summary_rows)} 个独立微调用户平均 Test Accuracy: {mean_acc:.4f}")
    print(f"汇总 CSV 已保存: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
