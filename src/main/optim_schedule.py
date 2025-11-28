import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import r2_score


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = 1e-4
        self.max_lr = 1.5e-4

    # def step_and_update_lr(self):
    #     "Step with the inner optimizer"
    #     self._update_learning_rate()
    #     self._optimizer.step()

    def step(self):  # 仅更新学习率
        self._update_learning_rate()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        # Linear warmup
        if self.n_current_steps <= self.n_warmup_steps:
            return (self.max_lr - self.init_lr) / self.n_warmup_steps * self.n_current_steps + self.init_lr
        # Inverse square root decay
        return self.max_lr * (self.n_warmup_steps ** 0.5) * (self.n_current_steps ** -0.5)

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None, reduction='mean', ignore_index=None):
        """
        gamma: Focusing parameter, default is 2.
        alpha: Weighting factor for each class, can be None or a list of length equal to number of classes.
        reduction: 'mean' or 'sum' to reduce the loss values.
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index


    def forward(self, inputs, targets):
        """
        inputs: Model predictions, shape (batch, seq_len, num_classes).
        targets: Ground truth labels, shape (batch, seq_len).
        """
        num_classes = inputs.size(-1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # (batch, seq_len, num_classes)

        # Apply softmax to get class probabilities
        probs = F.softmax(inputs, dim=-1)

        # Get the probabilities of the true classes
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (batch, seq_len)

        # Compute the focal loss
        loss = -((1 - p_t) ** self.gamma) * torch.log(p_t + 1e-10)  # Add epsilon to avoid log(0)

        if self.alpha is not None:
            alpha_t = (self.alpha * targets_one_hot).sum(dim=-1)  # (batch, seq_len)
            loss = alpha_t * loss

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            loss = loss * valid_mask.float()

        if self.reduction == 'mean':
            if self.ignore_index is not None:
                return loss.sum() / valid_mask.sum()  # Only consider valid elements in the mean
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
        
def cal_acc(pred : torch.Tensor,
            label : torch.Tensor,
            mask : torch.Tensor) -> tuple[int, int]:
    _pred = pred.argmax(dim=-1).flatten()
    _label = label.flatten()
    _mask = mask.flatten().bool()

    acc = (_pred.eq(_label) * _mask).sum().item()
    tot = _mask.sum().item()

    return acc, tot

def cal_acc_log(pred: torch.Tensor, 
            label: torch.Tensor,
            mask: torch.Tensor,
            seq_data: dict = None,      # 包含原始序列信息的字典
            sample_idx: int = None,     # 当前样本在batch中的索引
            print_threshold: float = 0.5, # 置信度阈值
            context_window: int = 5     # 上下文窗口大小（左右各显示n个位点）
           ) -> tuple[int, int]:
    """
    改进版准确率计算函数：
    1. 移除pos_start参数
    2. 直接使用原始位置数据
    3. 显示更多上下文信息
    """
    _pred = pred.argmax(dim=-1).flatten()
    _label = label.flatten()
    _mask = mask.flatten().bool()

    acc = (_pred.eq(_label) * _mask).sum().item()
    tot = _mask.sum().item()

    if seq_data is not None and sample_idx is not None:
        # 获取当前样本的原始数据
        batch_size, seq_len = seq_data['hap_seq'].shape
        pos_array = seq_data['pos'][sample_idx].cpu().numpy()  # [L]
        af_array = seq_data['af'][sample_idx].cpu().numpy()     # [L]
        hap_array = seq_data['hap_seq'][sample_idx].cpu().numpy() # [L]
        
        # 获取有效预测索引
        valid_indices = torch.where(_mask)[0].cpu().numpy()
        pred_1_mask = (_pred[_mask] == 1).cpu().numpy()
        
        # 打印预测为1的位点
        for idx in np.where(pred_1_mask)[0][:3]:  # 最多打印3个
            if idx >= len(pos_array):
                continue  # 防止越界
            
            global_pos = pos_array[idx]
            local_seq_idx = valid_indices[idx]
            
            # 获取上下文窗口（调整越界情况）
            start = max(0, local_seq_idx - context_window)
            end = min(len(hap_array), local_seq_idx + context_window + 1)
            
            print(f"""
            [预测为1的位点] 样本:{sample_idx}
            全局位置: {global_pos}
            局部索引: {local_seq_idx}
            上下文序列: {hap_array[start:end]}
            等位频率: {af_array[idx]:.4f}
            真实标签: {_label[_mask][idx].item()}
            """)

    return acc, tot


def cal_pr(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, num_classes: int):
    """
    新增：计算各类别的TP/FP/FN
    Args:
        pred: [B, L, C]
        label: [B, L]
        mask: [B, L]
        num_classes: 类别数量
    Returns:
        dict: {"tp":..., "fp":..., "fn":...}
    """
    _pred = pred.argmax(dim=-1).flatten()  # [N]
    _label = label.flatten()               # [N]
    _mask = mask.flatten().bool()          # [N]
    
    # 仅保留有效位置
    valid_pred = _pred[_mask]
    valid_label = _label[_mask]
    
    # 初始化统计字典
    stats = {
        "tp": torch.zeros(num_classes, dtype=torch.long),
        "fp": torch.zeros(num_classes, dtype=torch.long),
        "fn": torch.zeros(num_classes, dtype=torch.long)
    }
    
    # 逐类别统计
    for c in range(num_classes):
        tp = ((valid_pred == c) & (valid_label == c)).sum().item()
        fp = ((valid_pred == c) & (valid_label != c)).sum().item()
        fn = ((valid_pred != c) & (valid_label == c)).sum().item()
        
        stats["tp"][c] += tp
        stats["fp"][c] += fp
        stats["fn"][c] += fn
    
    return stats

