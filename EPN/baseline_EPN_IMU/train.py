import copy
import torch
from config import GRAD_CLIP


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        logits = out[1] if isinstance(out, tuple) else out
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
            out = model(x)
            logits = out[1] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="max"):
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

        if self.mode == "max":
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
            if self.counter >= self.patience:
                return True

        return False

    def load_best_model(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)
