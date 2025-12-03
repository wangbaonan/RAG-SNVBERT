"""
AF (Allele Frequency) Embedding Module

将连续的AF值 (0-1) 编码为高维向量，使用Fourier Features

理论基础:
- 类似BERT的Positional Embedding
- 类似NeRF的Positional Encoding
- Fourier features可以表达任意连续函数
"""

import torch
import torch.nn as nn
import math


class AFEmbedding(nn.Module):
    """
    Allele Frequency Embedding using Fourier Features

    将连续的AF值编码为与token embedding同等维度的向量

    Args:
        embed_size: Embedding维度 (通常与token embedding相同)
        num_basis: Fourier basis的数量 (越多表达能力越强)
        learnable_basis: 是否让basis frequencies可学习

    Example:
        >>> af_emb = AFEmbedding(embed_size=192, num_basis=32)
        >>> af = torch.tensor([[0.02, 0.45, 0.15]])  # [B=1, L=3]
        >>> emb = af_emb(af)  # [1, 3, 192]
    """

    def __init__(self,
                 embed_size: int = 192,
                 num_basis: int = 32,
                 learnable_basis: bool = True):
        super().__init__()
        self.embed_size = embed_size
        self.num_basis = num_basis

        if learnable_basis:
            # 可学习的basis frequencies
            # 初始化为log-spaced frequencies (覆盖不同尺度)
            init_freqs = torch.logspace(0, math.log10(100), num_basis)
            self.basis_freqs = nn.Parameter(init_freqs)
        else:
            # 固定的basis (类似NeRF)
            freqs = 2.0 ** torch.arange(num_basis, dtype=torch.float32)
            self.register_buffer('basis_freqs', freqs)

        # 将Fourier features投影到embed_size
        self.projection = nn.Sequential(
            nn.Linear(num_basis * 2, embed_size),  # sin + cos = 2 * num_basis
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, af: torch.Tensor) -> torch.Tensor:
        """
        Args:
            af: [B, L] - Allele frequency (0-1)

        Returns:
            [B, L, embed_size] - AF embedding
        """
        # 1. Expand AF with basis frequencies
        af_expanded = af.unsqueeze(-1) * self.basis_freqs  # [B, L, num_basis]

        # 2. Compute Fourier features (sin + cos)
        af_sin = torch.sin(2 * math.pi * af_expanded)  # [B, L, num_basis]
        af_cos = torch.cos(2 * math.pi * af_expanded)  # [B, L, num_basis]

        # 3. Concatenate sin and cos features
        af_features = torch.cat([af_sin, af_cos], dim=-1)  # [B, L, 2*num_basis]

        # 4. Project to embed_size
        af_emb = self.projection(af_features)  # [B, L, embed_size]

        return af_emb

    def visualize_encoding(self, af_values):
        """
        可视化不同AF值的编码 (用于调试和分析)

        Args:
            af_values: list or array of AF values

        Returns:
            numpy array of embeddings [len(af_values), embed_size]
        """
        with torch.no_grad():
            af_tensor = torch.tensor(af_values, dtype=torch.float32).unsqueeze(0)  # [1, L]
            embeddings = self.forward(af_tensor)  # [1, L, D]
            return embeddings.squeeze(0).cpu().numpy()  # [L, D]


class DualAFEmbedding(nn.Module):
    """
    处理两个AF: global_af和pop_af

    将两个AF分别编码后融合
    """

    def __init__(self,
                 embed_size: int = 192,
                 num_basis: int = 32,
                 learnable_basis: bool = True):
        super().__init__()

        self.global_af_emb = AFEmbedding(embed_size, num_basis, learnable_basis)
        self.pop_af_emb = AFEmbedding(embed_size, num_basis, learnable_basis)

        # 融合global和pop AF
        self.fusion = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU()
        )

    def forward(self, global_af: torch.Tensor, pop_af: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            global_af: [B, L] - Global allele frequency
            pop_af: [B, L] - Population-specific allele frequency (optional)

        Returns:
            [B, L, embed_size] - Fused AF embedding
        """
        global_emb = self.global_af_emb(global_af)  # [B, L, D]

        if pop_af is not None:
            pop_emb = self.pop_af_emb(pop_af)  # [B, L, D]
            # 融合两个AF
            fused = self.fusion(torch.cat([global_emb, pop_emb], dim=-1))  # [B, L, D]
            return fused
        else:
            return global_emb


class SimpleAFEmbedding(nn.Module):
    """
    简化版AF Embedding (如果Fourier Features太复杂)

    使用简单的MLP将AF编码到高维
    """

    def __init__(self, embed_size: int = 192):
        super().__init__()
        self.embed_size = embed_size

        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_size),
            nn.LayerNorm(embed_size)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, af: torch.Tensor) -> torch.Tensor:
        """
        Args:
            af: [B, L] - Allele frequency

        Returns:
            [B, L, embed_size]
        """
        af_expanded = af.unsqueeze(-1)  # [B, L, 1]
        af_emb = self.encoder(af_expanded)  # [B, L, embed_size]
        return af_emb
