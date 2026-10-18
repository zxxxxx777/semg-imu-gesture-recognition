import os
import sys
import csv
import random
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

from config import *
from data import MultiModalDataset
from models import FeatureFusionNet
from train import EarlyStopping, train_fusion_one_epoch, evaluate_fusion
from utils import load_subject_multimodal_with_rep, sliding_window_multimodal_by_segment


ALL_SUBJECTS = list(range(1, 21))
FOLD_TARGET_SUBJECTS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]

RESULT_ROOT = "/hdd/zhaozixuan/TEST_DB2/result/DB2_5fold_Fusion_BiResCrossAttn"
SUMMARY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DB2_5fold_Fusion_BiResCrossAttn_summary.csv",
)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(SEED)
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


def build_pretrain_loader(pretrain_subjects):
    print(f"\n[1/4] 正在加载预训练受试者数据: {pretrain_subjects}")
    pre_emg_list, pre_imu_list, pre_label_list, pre_rep_list = [], [], [], []

    for sub_id in pretrain_subjects:
        emg, imu, label, rep = load_subject_multimodal_with_rep(sub_id)
        pre_emg_list.append(emg)
        pre_imu_list.append(imu)
        pre_label_list.append(label)
        pre_rep_list.append(rep)

    print("计算全局 Z-Score 参数...")
    raw_all_pre_emg = np.vstack(pre_emg_list)
    raw_all_pre_imu = np.vstack(pre_imu_list)
    emg_mean = np.mean(raw_all_pre_emg, axis=0)
    emg_std = np.std(raw_all_pre_emg, axis=0) + 1e-8
    imu_mean = np.mean(raw_all_pre_imu, axis=0)
    imu_std = np.std(raw_all_pre_imu, axis=0) + 1e-8
    emg_ch = raw_all_pre_emg.shape[1]
    imu_ch = raw_all_pre_imu.shape[1]
    del raw_all_pre_emg, raw_all_pre_imu

    print("正在按受试者分别进行预训练数据切片...")
    X_pre_emg_list, X_pre_imu_list, Y_pre_list = [], [], []
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
        Y_pre_list.append(Yw)

    X_pre_emg = np.concatenate(X_pre_emg_list, axis=0)
    X_pre_imu = np.concatenate(X_pre_imu_list, axis=0)
    Y_pre = np.concatenate(Y_pre_list, axis=0)
    del pre_emg_list, pre_imu_list, pre_label_list, pre_rep_list
    del X_pre_emg_list, X_pre_imu_list, Y_pre_list

    pretrain_loader = make_loader(
        MultiModalDataset(X_pre_emg, X_pre_imu, Y_pre),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    sample_count = len(Y_pre)
    print(f"预训练集构建完成: {sample_count} 个窗口")

    del X_pre_emg, X_pre_imu, Y_pre

    return pretrain_loader, emg_mean, emg_std, imu_mean, imu_std, emg_ch, imu_ch, sample_count


def build_finetune_loaders(target_subject, emg_mean, emg_std, imu_mean, imu_std):
    print(f"\n加载目标被试 Subject {target_subject} 数据...")
    emg_target, imu_target, label_target, rep_target = load_subject_multimodal_with_rep(target_subject)
    emg_target = (emg_target - emg_mean) / emg_std
    imu_target = (imu_target - imu_mean) / imu_std

    train_mask = np.isin(rep_target, TRAIN_REPS)
    test_mask = np.isin(rep_target, TEST_REPS)

    X_ft_train_emg, X_ft_train_imu, Y_ft_train = sliding_window_multimodal_by_segment(
        emg_target[train_mask],
        imu_target[train_mask],
        label_target[train_mask],
        rep_target[train_mask],
        step_size=FINETUNE_STEP_SIZE,
    )
    ft_train_loader = make_loader(
        MultiModalDataset(X_ft_train_emg, X_ft_train_imu, Y_ft_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    X_ft_test_emg, X_ft_test_imu, Y_ft_test = sliding_window_multimodal_by_segment(
        emg_target[test_mask],
        imu_target[test_mask],
        label_target[test_mask],
        rep_target[test_mask],
        step_size=FINETUNE_STEP_SIZE,
    )
    ft_test_loader = make_loader(
        MultiModalDataset(X_ft_test_emg, X_ft_test_imu, Y_ft_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    train_count = len(Y_ft_train)
    test_count = len(Y_ft_test)
    print(f"微调训练集: {train_count} 窗口 | 测试集: {test_count} 窗口")

    del emg_target, imu_target, label_target, rep_target
    del X_ft_train_emg, X_ft_train_imu, Y_ft_train
    del X_ft_test_emg, X_ft_test_imu, Y_ft_test

    return ft_train_loader, ft_test_loader, train_count, test_count


def create_fusion_model(emg_ch, imu_ch, device):
    return FeatureFusionNet(
        emg_ch=emg_ch,
        imu_ch=imu_ch,
        num_classes=NUM_CLASSES,
        lstm_hidden=LSTM_HIDDEN,
    ).to(device)


def pretrain_one_fold(fold_idx, pretrain_subjects, fold_save_dir, device):
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
        np.savez(
            pretrain_log_path,
            pretrain_loss=np.array(pretrain_loss_log),
            pretrain_subjects=np.array(pretrain_subjects),
            sample_count=sample_count,
            emg_mean=emg_mean,
            emg_std=emg_std,
            imu_mean=imu_mean,
            imu_std=imu_std,
            loaded_from_cache=True,
        )
    else:
        pretrain_loader, emg_mean, emg_std, imu_mean, imu_std, emg_ch, imu_ch, sample_count = build_pretrain_loader(
            pretrain_subjects
        )

        print("\n[2/4] 初始化双向残差交叉注意力互学习网络...")
        model = create_fusion_model(emg_ch, imu_ch, device)
        pretrain_criterion = nn.CrossEntropyLoss(label_smoothing=PRETRAIN_LABEL_SMOOTHING)

        print(f"\n[3/4] Fold {fold_idx} 开始 Phase 1: 跨人预训练 ({PRETRAIN_EPOCHS} Epochs)")
        print(f"预训练 Subjects: {pretrain_subjects}")
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
            print(f"[Pre-train {epoch+1}/{PRETRAIN_EPOCHS}] Loss:{train_loss:.4f} | LR:{optimizer_pre.param_groups[0]['lr']:.6f}")

        torch.save(
            {
                "model_state": model.state_dict(),
                "emg_mean": emg_mean,
                "emg_std": emg_std,
                "imu_mean": imu_mean,
                "imu_std": imu_std,
                "emg_ch": emg_ch,
                "imu_ch": imu_ch,
                "pretrain_subjects": pretrain_subjects,
                "sample_count": sample_count,
            },
            pretrain_cache_path,
        )
        np.savez(
            pretrain_log_path,
            pretrain_loss=np.array(pretrain_loss_log),
            pretrain_subjects=np.array(pretrain_subjects),
            sample_count=sample_count,
            emg_mean=emg_mean,
            emg_std=emg_std,
            imu_mean=imu_mean,
            imu_std=imu_std,
            loaded_from_cache=False,
        )
        print(f"已保存 Fold {fold_idx} 预训练权重: {pretrain_cache_path}")

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
        "sample_count": sample_count,
        "pretrain_subjects": pretrain_subjects,
    }


def finetune_one_subject(fold_idx, target_subject, pretrained_info, subject_save_dir, device):
    ft_train_loader, ft_test_loader, train_count, test_count = build_finetune_loaders(
        target_subject,
        pretrained_info["emg_mean"],
        pretrained_info["emg_std"],
        pretrained_info["imu_mean"],
        pretrained_info["imu_std"],
    )

    print("\n[4/4] 初始化并加载预训练权重，开始目标被试微调...")
    model = create_fusion_model(pretrained_info["emg_ch"], pretrained_info["imu_ch"], device)
    cache = load_pretrain_cache(pretrained_info["pretrained_model_path"])
    model.load_state_dict(cache["model_state"])

    finetune_criterion = nn.CrossEntropyLoss(label_smoothing=FINETUNE_LABEL_SMOOTHING)
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS, eta_min=1e-6)
    early_stopping_ft = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")

    ft_loss_log, ft_test_acc_log, gate_emg_log, gate_imu_log = [], [], [], []
    best_gate_emg = 0.0
    best_gate_imu = 0.0

    print(f"Fold {fold_idx} | 训练: Sub {target_subject} Rep {TRAIN_REPS} | 测试: Sub {target_subject} Rep {TEST_REPS}")
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
            f"[Fine-tune {epoch+1}/{EPOCHS}] Loss:{train_loss:.4f} | Test Acc:{test_acc:.4f} | "
            f"LR:{optimizer_ft.param_groups[0]['lr']:.6f} | GateE:{gate_emg:.3f} | GateI:{gate_imu:.3f}"
        )

        should_stop = early_stopping_ft(test_acc, model, epoch)
        if early_stopping_ft.best_epoch == epoch:
            best_gate_emg = gate_emg
            best_gate_imu = gate_imu
        if should_stop:
            print(f"微调早停触发! 最佳验证准确率: {early_stopping_ft.best_score:.4f} (Epoch {early_stopping_ft.best_epoch+1})")
            break

    best_model_path = os.path.join(subject_save_dir, "best_bires_crossattn_mutual_model.pth")
    finetune_log_path = os.path.join(subject_save_dir, "Fusion_Finetune_log.npz")
    torch.save(early_stopping_ft.best_model, best_model_path)
    np.savez(
        finetune_log_path,
        ft_loss=np.array(ft_loss_log),
        ft_test_acc=np.array(ft_test_acc_log),
        gate_emg=np.array(gate_emg_log),
        gate_imu=np.array(gate_imu_log),
        train_reps=np.array(TRAIN_REPS),
        test_reps=np.array(TEST_REPS),
        train_count=train_count,
        test_count=test_count,
        best_acc=early_stopping_ft.best_score,
        best_epoch=early_stopping_ft.best_epoch + 1,
        best_gate_emg=best_gate_emg,
        best_gate_imu=best_gate_imu,
    )

    result = {
        "fold": fold_idx,
        "target_subject": target_subject,
        "train_reps": "-".join(map(str, TRAIN_REPS)),
        "test_reps": "-".join(map(str, TEST_REPS)),
        "pretrain_subjects": "-".join(map(str, pretrained_info["pretrain_subjects"])),
        "pretrain_windows": pretrained_info["sample_count"],
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

    print(f"\n训练结束。Fold {fold_idx} Subject {target_subject} 最佳测试准确率: {early_stopping_ft.best_score:.4f}")

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
    set_seed(SEED)
    device = torch.device(DEVICE)
    os.makedirs(RESULT_ROOT, exist_ok=True)

    print("========== DB2 双向残差交叉注意力互学习融合网络五折预训练+独立微调 ==========")
    print(f"使用设备: {DEVICE}")
    print(f"随机种子: {SEED}")
    print(
        f"FineTune KL:{FINETUNE_BETA_KL} | CONS:{FINETUNE_BETA_CONS} | "
        f"RelAlign:{FINETUNE_BETA_REL_ALIGN} | ModDrop:{MODALITY_DROPOUT_P}"
    )
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
    print(f"\nDB2 五折流程全部完成！20 个独立微调结果平均 Test Accuracy: {mean_acc:.4f}")
    print(f"汇总 CSV 已保存: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
