import torch
import copy
from config import GRAD_CLIP


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() #将模型设置为训练模式
    total_loss = 0

    for x, y in loader:   #从数据加载器中获取一个批次的数据
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()   #将优化器的梯度清零，准备计算新的梯度
        out = model(x)
        
        #logits是模型的输出，可能是一个元组（features, logits）或者直接是logits。
        if isinstance(out, tuple):#判断out是否是一个元组，如果是，则取第二个元素作为logits
            _, logits = out 
        else:
            logits = out

        loss = criterion(logits, y)
        loss.backward()    #反向传播，计算每个参数的梯度值
        
        #梯度裁剪，防止梯度过大。model.parameters返回模型的所有参数，torch.nn.utils.clip_grad_norm_将参数的梯度裁剪到最大范数为GRAD_CLIP。
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()   #根据梯度更新模型参数

        total_loss += loss.item() #loss是张量，.item()将其转换为标量
    
    #len(loader)不是样本数，而是批次数。因为total_loss是所有批次的损失总和
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():    #禁用梯度计算，评估阶段不需要更新模型参数
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            
            if isinstance(out, tuple):
                _, logits = out
            else:
                logits = out
            
            loss = criterion(logits, y)

            total_loss += loss.item()

            pred = logits.argmax(1) #获取预测结果，argmax(1)表示在最后一个维度上取最大值的索引
            correct += (pred == y).sum().item() #统计预测正确的样本数。pred == y ：布尔张量，预测正确为 True。.item() ：转换为 Python 标量。
    
    #len(loader.dataset)是样本总数
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
            if self.counter >= self.patience:
                return True

        return False

    def load_best_model(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)