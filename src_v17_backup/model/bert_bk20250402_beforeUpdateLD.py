import torch.nn as nn
import torch
import numpy as np

from typing import Optional

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .fusion import EmbeddingFusionModule, CrossAttentionFusion, ConcatFusion, FixedConcatFusion, RareVariantAwareFusion, EnhancedRareVariantFusion


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
    """改进版RAG-BERT模型"""
    def __init__(self, vocab_size, dims=512, n_layers=12, attn_heads=16, dropout=0.1):
        super().__init__(vocab_size, dims, n_layers, attn_heads, dropout)
        
        # 替换融合模块
        self.rag_fusion_h1 = EnhancedRareVariantFusion(dims)
        self.rag_fusion_h2 = EnhancedRareVariantFusion(dims)

    def encode_rag_segments(self, rag_segs, pos, af):
        """显存优化版参考编码"""
        B, K, L = rag_segs.size()
        chunk_size = max(1, 512 // L)  # 自动分块
        
        encoded_chunks = []
        for i in range(0, K, chunk_size):
            chunk = rag_segs[:, i:i+chunk_size]
            Bc, Kc, Lc = chunk.size()
            
            # 展开批次
            chunk_flat = chunk.reshape(-1, Lc)
            pos_exp = pos.unsqueeze(1).expand(-1, Kc, -1).reshape(-1, Lc)
            af_exp = af.unsqueeze(1).expand(-1, Kc, -1).reshape(-1, Lc)
            
            # 编码过程
            emb = self.embedding(chunk_flat)
            emb = self.emb_fusion(emb, pos_exp, af_exp)
            for t in self.transformer_blocks:
                if self.training:
                    emb = torch.utils.checkpoint.checkpoint(t, emb, use_reentrant=False)
                else:
                    emb = t(emb)

            
            encoded_chunks.append(emb.view(B, Kc, Lc, -1))
        
        return torch.cat(encoded_chunks, dim=1)

    def forward(self, x: dict) -> tuple:
        # 修改输入包含pop_af
        h1, h2, h1_ori, h2_ori = super().forward(x)
        
        # 参考编码
        rag_h1 = self.encode_rag_segments(x['rag_seg_h1'], x['pos'], x['af'])
        rag_h2 = self.encode_rag_segments(x['rag_seg_h2'], x['pos'], x['af'])
        
        # 增强融合
        h1_fused = self.rag_fusion_h1(h1, rag_h1, x['af'], x['af_p'])
        h2_fused = self.rag_fusion_h2(h2, rag_h2, x['af'], x['af_p'])
        
        return h1_fused, h2_fused, h1_ori, h2_ori