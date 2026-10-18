import copy
import torch
import torch.nn.functional as F
from config import GRAD_CLIP, MUTUAL_KL_TEMPERATURE


def symmetric_kl(student_logits, teacher_logits, temperature=MUTUAL_KL_TEMPERATURE):
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def gate_sparsity_loss(out):
    return out["gate_emg_map"].abs().mean() + out["gate_imu_map"].abs().mean()


def cross_modal_relational_alignment_loss(z_emg, z_imu):
    if z_emg.size(0) < 2:
        return z_emg.new_zeros(())

    z_emg = F.normalize(z_emg, dim=1)
    z_imu = F.normalize(z_imu, dim=1)
    sim_emg = z_emg.matmul(z_emg.t())
    sim_imu = z_imu.matmul(z_imu.t())
    off_diag = ~torch.eye(z_emg.size(0), dtype=torch.bool, device=z_emg.device)

    return F.mse_loss(sim_emg[off_diag], sim_imu[off_diag])


def train_fusion_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    alpha=0.3,
    beta_enh=0.2,
    beta_kl=0.05,
    beta_cons=0.02,
    beta_gate=0.0,
    beta_rel_align=0.0,
):
    model.train()
    total_loss = 0
    for x_emg, x_imu, y in loader:
        x_emg, x_imu, y = x_emg.to(device), x_imu.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_emg, x_imu)

        loss_fusion = criterion(out["fusion_logits"], y)
        loss_aux = criterion(out["emg_logits"], y) + criterion(out["imu_logits"], y)
        loss_enh = criterion(out["emg_enh_logits"], y) + criterion(out["imu_enh_logits"], y)

        # 双向跨模态互学习：EMG 增强分支学习 IMU 原分支预测，IMU 增强分支学习 EMG 原分支预测。
        loss_kl = (
            symmetric_kl(out["emg_enh_logits"], out["imu_logits"])
            + symmetric_kl(out["imu_enh_logits"], out["emg_logits"])
        )

        # 增强后不能偏离本模态已学到的类别判别太远，避免互学把强模态带坏。
        loss_cons = (
            symmetric_kl(out["emg_enh_logits"], out["emg_logits"])
            + symmetric_kl(out["imu_enh_logits"], out["imu_logits"])
        )
        loss_gate = gate_sparsity_loss(out) if beta_gate > 0 else out["fusion_logits"].new_zeros(())
        loss_rel_align = out["fusion_logits"].new_zeros(())
        if beta_rel_align > 0:
            loss_rel_align = cross_modal_relational_alignment_loss(out["align_emg_enh"], out["align_imu_enh"])

        loss = (
            loss_fusion
            + alpha * loss_aux
            + beta_enh * loss_enh
            + beta_kl * loss_kl
            + beta_cons * loss_cons
            + beta_gate * loss_gate
            + beta_rel_align * loss_rel_align
        )
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
    gate_emg_total = 0
    gate_imu_total = 0
    num_batches = 0
    with torch.no_grad():
        for x_emg, x_imu, y in loader:
            x_emg, x_imu, y = x_emg.to(device), x_imu.to(device), y.to(device)
            out = model(x_emg, x_imu)
            loss = criterion(out["fusion_logits"], y)
            total_loss += loss.item()
            pred = out["fusion_logits"].argmax(1)
            correct += (pred == y).sum().item()
            gate_emg_total += out["gate_emg"].item()
            gate_imu_total += out["gate_imu"].item()
            num_batches += 1
    acc = correct / len(loader.dataset)
    gate_emg = gate_emg_total / max(num_batches, 1)
    gate_imu = gate_imu_total / max(num_batches, 1)
    return total_loss / len(loader), acc, gate_emg, gate_imu


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        _, logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.best_epoch = 0

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
