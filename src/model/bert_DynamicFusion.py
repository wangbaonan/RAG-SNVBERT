import torch.nn as nn
import torch
import numpy as np

from typing import Optional

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .fusion import EmbeddingFusionModule, CrossAttentionFusion, ConcatFusion, FixedConcatFusion, RareVariantAwareFusion, EnhancedRareVariantFusion, DynamicGeneFusion


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
    """改进版RAG-BERT（增加共享层机制）"""
    def __init__(self, vocab_size, dims=512, n_layers=12, attn_heads=8, dropout=0.1,
                shared_layers=4):  # 新增共享层参数
        super().__init__(vocab_size, dims, n_layers, attn_heads, dropout)
        
        # 划分共享层与专用层 ==============================
        # 共享层定义（前shared_layers层）
        self.shared_transformer = nn.ModuleList(
            self.transformer_blocks[:shared_layers]
        )
        # RAG专用层定义（剩余层）
        self.rag_transformer = nn.ModuleList(
            self.transformer_blocks[shared_layers:]
        )
        # ================================================
        
        # 初始化融合模块
        self.rag_fusion_h1 = DynamicGeneFusion(dims)
        self.rag_fusion_h2 = DynamicGeneFusion(dims)
        
        # 冻结共享层梯度（可选）
        for param in self.shared_transformer.parameters():
            param.requires_grad_(False)

    def encode_rag_segments(self, rag_segs, pos, af):
        B, K, L = rag_segs.size()
        
        # 展平处理保持兼容
        flat_segs = rag_segs.view(B*K, L)  # L是序列长度
        
        # 共享编码过程（关键修正）
        emb = self.embedding(flat_segs)  # [B*K, L, D]
        pos_emb = self.embedding.position(flat_segs)  # 从embedding模块获取位置编码
        
        # 拼接特征
        fused = emb + pos_emb
        
        # Transformer处理
        for layer in self.shared_transformer:
            fused = layer(fused)
            
        # 恢复形状时保持维度正确
        return fused.view(B, K, L, -1)  # [B, K, L, D]

    def forward(self, x):
        # 主路径处理
        h1, h2, h1_ori, h2_ori = super().forward(x)
        
        # 参考路径处理（带维度转换）
        rag_h1 = self.encode_rag_segments(x['rag_seg_h1'], x['pos'], x['af'])
        rag_h2 = self.encode_rag_segments(x['rag_seg_h2'], x['pos'], x['af'])
        
        # 动态融合（严格输入校验）
        assert rag_h1.size(-1) == h1.size(-1), "主参考特征维度不匹配"
        h1_fused = self.rag_fusion_h1(h1, rag_h1, x['af'], x['af_p'])
        h2_fused = self.rag_fusion_h2(h2, rag_h2, x['af'], x['af_p'])
        
        return h1_fused, h2_fused, h1_ori, h2_ori