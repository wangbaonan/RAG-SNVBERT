import torch
import torch.nn as nn

from typing import Optional

# from .token import TokenEmbedding
from .position import PositionalEmbedding

class GeneAwareEmbedding(nn.Embedding):
    """基因特化嵌入（兼容您现有token映射）"""
    def __init__(self, vocab_size, embed_size, padding_idx=0):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)
        self._init_gene_weights(vocab_size)

    def _init_gene_weights(self, vocab_size):
        with torch.no_grad():
            # 确保词表包含所有必需token
            if vocab_size < 7:
                raise ValueError(f"词表大小需≥7（当前{vocab_size}），需包含0-6索引")
            
            # 特殊token初始化
            nn.init.constant_(self.weight[0], 0.0)   # pad_index=0
            nn.init.normal_(self.weight[1], 0.0, 0.02)  # unk_index=1
            nn.init.normal_(self.weight[2], 0.0, 0.01)  # sos_index=2
            nn.init.normal_(self.weight[3], 0.0, 0.01)  # eos_index=3
            nn.init.uniform_(self.weight[4], -0.01, 0.01)  # mask_index=4
            
            # 基因型token特殊初始化
            nn.init.constant_(self.weight[5], 0.0)   # 原始0→5，多数等位基因
            nn.init.normal_(self.weight[6], mean=1.0, std=0.1)  # 原始1→6，少数等位基因


class BERTEmbedding(nn.Module):
    """整合改进的嵌入模块"""
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        
        # 基因特化嵌入
        self.tokenizer = GeneAwareEmbedding(
            vocab_size=vocab_size,
            embed_size=embed_size,
            padding_idx=0
        )
        
        # 位置编码
        self.position = PositionalEmbedding(embed_size)
        
        # 自适应缩放（符合Transformer惯例）
        self.scale = torch.sqrt(torch.tensor(embed_size, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
        
        # 兼容性验证
        self._validate_vocab_size(vocab_size)

    def _validate_vocab_size(self, vocab_size):
        if vocab_size < 7:
            raise ValueError(
                f"词表大小至少需要7（当前{vocab_size}），索引分配："
                "0:pad,1:unk,2:sos,3:eos,4:mask,5:0,6:1"
            )

    def forward(self, seq, pos=True):
        # 添加维度校验
        assert seq.dim() == 2, f"输入应为2D [B,L]，当前为{seq.dim()}D"
        
        # Token嵌入
        token_emb = self.tokenizer(seq) * self.scale  # [B,L,D]
        
        # 位置编码（带序列长度校验）
        if pos:
            max_len = self.position.pe.size(1)
            input_len = seq.size(1)
            assert input_len <= max_len, f"输入长度{input_len}超过最大支持{max_len}"
            pos_emb = self.position(seq)  # [B,L,D]
            token_emb += pos_emb
        
        return self.dropout(token_emb)