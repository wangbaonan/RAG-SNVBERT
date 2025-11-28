import torch.nn as nn
import torch

import math

MAX_SEQ_LEN = 1030


class PositionalEmbedding(nn.Module):
    """改进版位置编码（带基因序列优化）"""
    def __init__(self, dims: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.dims = dims
        
        # 生成基础位置编码
        pe = torch.zeros(max_len, dims)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dims, 2).float() * (-math.log(10000.0) / dims))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 基因序列优化：下三角掩码防止位置泄漏
        self.register_buffer('pe', pe.unsqueeze(0))
        self.register_buffer('tril_mask', torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        # 严格维度控制
        assert x.dim() == 2, "输入应为2D张量 [B,L]"
        seq_len = x.size(1)
        
        # 带掩码的安全位置编码
        if seq_len > self.pe.size(1):
            raise ValueError(f"序列长度{seq_len}超过预计算最大值{self.pe.size(1)}")
            
        # 应用三角形掩码防止位置泄漏
        return torch.matmul(self.tril_mask[:seq_len, :seq_len], self.pe[:, :seq_len])

