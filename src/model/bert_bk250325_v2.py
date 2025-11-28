import torch.nn as nn
import torch
import numpy as np

from typing import Optional

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .fusion import EmbeddingFusionModule, CrossAttentionFusion, ConcatFusion, FixedConcatFusion, RareVariantAwareFusion


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
        self.rag_fusion_h1 = RareVariantAwareFusion(dims)
        self.rag_fusion_h2 = RareVariantAwareFusion(dims)

    def encode_rag_segments(self, rag_segs: torch.Tensor, pos: torch.Tensor, af: torch.Tensor) -> torch.Tensor:
        """显存优化的参考序列编码"""
        B, K, L = rag_segs.size()
        device = rag_segs.device
        
        # 分块处理防止显存溢出
        chunk_size = max(1, 512 // L)  # 自动调整分块大小
        encoded_chunks = []
        
        for k in range(0, K, chunk_size):
            seg_chunk = rag_segs[:, k:k+chunk_size]
            chunk_size_actual = seg_chunk.size(1)
            
            # 展平处理
            rag_flat = seg_chunk.reshape(B*chunk_size_actual, L)
            pos_expanded = pos.unsqueeze(1).expand(-1, chunk_size_actual, -1).reshape(B*chunk_size_actual, L)
            af_expanded = af.unsqueeze(1).expand(-1, chunk_size_actual, -1).reshape(B*chunk_size_actual, L)
            
            # 编码过程
            rag_emb = self.embedding(rag_flat)
            rag_emb = self.emb_fusion(rag_emb, pos_expanded, af_expanded)
            for transformer in self.transformer_blocks:
                rag_emb = transformer(rag_emb)
            
            encoded_chunks.append(rag_emb.view(B, chunk_size_actual, L, -1))
        
        return torch.cat(encoded_chunks, dim=1)

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
        h1_fused = self.rag_fusion_h1(h1, rag_h1, x['af'])  # [B, L, D]
        h2_fused = self.rag_fusion_h2(h2, rag_h2, x['af'])
        
        return h1_fused, h2_fused, h1_origin, h2_origin