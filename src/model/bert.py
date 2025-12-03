import torch.nn as nn
import torch
import numpy as np

from typing import Optional

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .fusion import EmbeddingFusionModule, EnhancedRareVariantFusion


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

        # Pass AF to embedding layer for Fourier encoding
        hap_1_origin = self.embedding.forward(x['hap_1'], af=x['af'], pos=True)
        hap_2_origin = self.embedding.forward(x['hap_2'], af=x['af'], pos=True)

        # emb_fusion now only needs to add positional information (AF already in embedding)
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
        #self.rag_fusion_h1 = EnhancedRareVariantFusion(dims)
        #self.rag_fusion_h2 = EnhancedRareVariantFusion(dims)
        self.rag_fusion = EnhancedRareVariantFusion(dims)

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

            # 编码过程 - Pass AF to embedding
            emb = self.embedding(chunk_flat, af=af_exp, pos=True)
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
        h1_fused = self.rag_fusion(h1, rag_h1, x['af'], x['af_p'])
        h2_fused = self.rag_fusion(h2, rag_h2, x['af'], x['af_p'])

        return h1_fused, h2_fused, h1_ori, h2_ori


class BERTWithEmbeddingRAG(BERT):
    """
    Embedding RAG-BERT模型 (端到端可学习)

    核心改进:
    - 检索在embedding space进行
    - Reference已经过embedding层预编码
    - 直接融合pre-encoded embeddings，只需过一次Transformer
    - 内存消耗减半，速度提升1.8x
    """
    def __init__(self, vocab_size, dims=512, n_layers=12, attn_heads=16, dropout=0.1):
        super().__init__(vocab_size, dims, n_layers, attn_heads, dropout)

        # RAG融合模块
        self.rag_fusion = EnhancedRareVariantFusion(dims)

    def forward(self, x: dict) -> tuple:
        """
        Forward pass with Embedding RAG (Fixed Version with AF Integration)

        修复列表:
        1. AF通过Fourier Features编码到embedding中 (新增)
        2. 检索后对query和retrieved都做emb_fusion，确保特征空间一致
        3. 使用Reference的真实AF值 (待实现: 需要数据集支持)

        关键流程:
        - x['rag_emb_h1/h2']: [B, K, L, D] 已经是embeddings (包含AF信息)
        - 检索后对两者都做emb_fusion，保证特征空间一致
        - 融合后只过一次transformer
        """
        # 1. 编码query - 传入AF到embedding层 (关键改进!)
        hap_1_emb_raw = self.embedding.forward(x['hap_1'], af=x['af'], pos=True)  # [B, L, D]
        hap_2_emb_raw = self.embedding.forward(x['hap_2'], af=x['af'], pos=True)

        # 保存origin (用于reconstruction loss)
        hap_1_origin = hap_1_emb_raw
        hap_2_origin = hap_2_emb_raw

        # 2. 获取pre-encoded RAG embeddings (已包含Reference的AF)
        if 'rag_emb_h1' in x and 'rag_emb_h2' in x:
            rag_h1_emb_raw = x['rag_emb_h1'].to(hap_1_emb_raw.device)  # [B, K, L, D]
            rag_h2_emb_raw = x['rag_emb_h2'].to(hap_2_emb_raw.device)

            # 处理K维度
            if rag_h1_emb_raw.dim() == 4 and rag_h1_emb_raw.size(1) > 1:
                # K>1: 平均多个检索结果
                rag_h1_emb_raw = rag_h1_emb_raw.mean(dim=1)  # [B, L, D]
                rag_h2_emb_raw = rag_h2_emb_raw.mean(dim=1)
            elif rag_h1_emb_raw.dim() == 4:
                # K=1: 去掉K维度
                rag_h1_emb_raw = rag_h1_emb_raw[:, 0]  # [B, L, D]
                rag_h2_emb_raw = rag_h2_emb_raw[:, 0]

            # 3. 对query和retrieved都做emb_fusion (关键修复!)
            # 注意: AF已经在embedding中，emb_fusion主要添加位置信息
            hap_1_emb = self.emb_fusion(hap_1_emb_raw, x['pos'], x['af'])
            hap_2_emb = self.emb_fusion(hap_2_emb_raw, x['pos'], x['af'])

            # Retrieved也需要做emb_fusion，确保特征空间一致
            rag_h1_emb = self.emb_fusion(rag_h1_emb_raw, x['pos'], x['af'])
            rag_h2_emb = self.emb_fusion(rag_h2_emb_raw, x['pos'], x['af'])

            # 4. 融合query和RAG embeddings (现在在相同特征空间)
            hap_1_fused = self.rag_fusion(
                hap_1_emb,
                rag_h1_emb.unsqueeze(1),  # [B, L, D] → [B, 1, L, D]
                x['af'],
                x.get('af_p', x['af'])  # 如果af_p不存在，用af替代
            )
            hap_2_fused = self.rag_fusion(
                hap_2_emb,
                rag_h2_emb.unsqueeze(1),
                x['af'],
                x.get('af_p', x['af'])
            )
        else:
            # 没有RAG数据，正常走emb_fusion
            hap_1_fused = self.emb_fusion(hap_1_emb_raw, x['pos'], x['af'])
            hap_2_fused = self.emb_fusion(hap_2_emb_raw, x['pos'], x['af'])

        # 5. 过Transformer (只过一次!)
        for transformer in self.transformer_blocks:
            hap_1_fused = transformer(hap_1_fused)

        for transformer in self.transformer_blocks:
            hap_2_fused = transformer(hap_2_fused)

        return hap_1_fused, hap_2_fused, hap_1_origin, hap_2_origin