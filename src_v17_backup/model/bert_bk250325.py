import torch.nn as nn
import torch
import numpy as np

from typing import Optional

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .fusion import EmbeddingFusionModule, CrossAttentionFusion, ConcatFusion, FixedConcatFusion


class BERT(nn.Module):

    def __init__(self,
                 vocab_size : int,
                 dims : int = 512,
                 n_layers : int = 12,
                 attn_heads : int = 16,
                 dropout : float = 0.1,
                 ):
        """

        Args:

            vocab_size.

            dims: BERT model hidden size.

            n_layers: numbers of Transformer Blocks.

            attn_heads: number of attention heads.

            dropout: dropout rate.
        """

        super().__init__()
        self.dims = dims
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = dims * 4
        
        self.vocab_size = vocab_size

        self.embedding = BERTEmbedding(vocab_size=vocab_size,
                                       embed_size=dims,
                                       dropout=dropout)
        
        self.emb_fusion = EmbeddingFusionModule(emb_size=dims)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dims=dims, attn_heads=attn_heads, feed_forward_hidden=self.feed_forward_hidden, dropout=dropout)
             for _ in range(self.n_layers)]
        )


    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assume input x is a DICT instance, and its values have shape like (batch_size, sequence_len).
        """

        hap_1_origin = self.embedding.forward(x['hap_1'])
        hap_2_origin = self.embedding.forward(x['hap_2'])

        hap_1 = self.emb_fusion(hap_1_origin, x['pos'], x['af'])
        hap_2 = self.emb_fusion(hap_2_origin, x['pos'], x['af'])
        
        for transformer in self.transformer_blocks:
            hap_1  = transformer(hap_1)

        for transformer in self.transformer_blocks:
            hap_2  = transformer(hap_2)

        return hap_1, hap_2, hap_1_origin, hap_2_origin

class BERTWithRAG(BERT):
    def __init__(self, 
                 vocab_size: int,
                 dims: int = 512,
                 n_layers: int = 12,
                 attn_heads: int = 16,
                 dropout: float = 0.1):
        """
        扩展后的BERT模型，支持RAG融合
        
        参数说明：
        - vocab_size: 词表大小（与原始BERT相同）
        - dims: 隐藏层维度（需与原始BERT一致）
        - 其他参数与原始BERT保持一致
        """
        # 初始化父类
        super().__init__(vocab_size, dims, n_layers, attn_heads, dropout)
        
        # 新增RAG相关模块
        self.rag_fusion_h1 = FixedConcatFusion(dims)
        self.rag_fusion_h2 = FixedConcatFusion(dims)

    def encode_rag_segments(self, rag_segs: torch.Tensor, pos: torch.Tensor, af: torch.Tensor) -> torch.Tensor:
        """
        编码参考序列的核心方法
        
        输入形状：
        - rag_segs: [batch_size, K, seq_len] 
        - pos:       [batch_size, seq_len]
        - af:        [batch_size, seq_len]
        
        输出形状：
        - [batch_size, K, seq_len, dims]
        """
        B, K, L = rag_segs.size()
        
        # 展平批次和K维度
        rag_flat = rag_segs.view(B*K, L)  # [B*K, L]
        
        # 生成广播后的位置和频率特征
        pos_expanded = pos.unsqueeze(1).expand(-1, K, -1).reshape(B*K, L)  # [B*K, L]
        af_expanded = af.unsqueeze(1).expand(-1, K, -1).reshape(B*K, L)    # [B*K, L]
        
        # 通过共享编码层
        rag_emb = self.embedding(rag_flat)  # [B*K, L, D]
        rag_emb = self.emb_fusion(rag_emb, pos_expanded, af_expanded)
        
        # 逐层处理Transformer
        for transformer in self.transformer_blocks:
            rag_emb = transformer(rag_emb)
            
        # 恢复原始维度
        return rag_emb.view(B, K, L, -1)  # [B, K, L, D]

    def forward(self, x: dict) -> tuple:
        """
        重写前向传播流程
        
        输入要求：
        x = {
            'hap_1':       [B, L], 
            'hap_2':       [B, L],
            'rag_seg_h1':  [B, K, L],  # 新增字段
            'rag_seg_h2':  [B, K, L],  # 新增字段
            'pos':         [B, L],
            'af':          [B, L]
        }
        """
        # ---------------------------
        # Step 1: 原始单体型编码
        # ---------------------------
        # [B, L] => [B, L, D]
        h1_origin = self.embedding(x['hap_1'])
        h2_origin = self.embedding(x['hap_2'])
        
        # 特征融合
        h1 = self.emb_fusion(h1_origin, x['pos'], x['af'])
        h2 = self.emb_fusion(h2_origin, x['pos'], x['af'])
        
        # Transformer编码
        for transformer in self.transformer_blocks:
            h1 = transformer(h1)
            h2 = transformer(h2)
            
        # ---------------------------
        # Step 2: 参考序列处理
        # ---------------------------
        # [B, K, L] => [B, K, L, D]
        rag_h1 = self.encode_rag_segments(x['rag_seg_h1'], x['pos'], x['af'])
        rag_h2 = self.encode_rag_segments(x['rag_seg_h2'], x['pos'], x['af'])
        
        # ---------------------------
        # Step 3: 跨注意力融合
        # ---------------------------
        h1_fused = self.rag_fusion_h1(h1, rag_h1)  # [B, L, D]
        h2_fused = self.rag_fusion_h2(h2, rag_h2)
        
        return h1_fused, h2_fused, h1_origin, h2_origin