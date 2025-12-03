import torch
import torch.nn as nn

from typing import Optional

# from .token import TokenEmbedding
from .position import PositionalEmbedding
from .af_embedding import AFEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding with integrated AF (Allele Frequency) information

    组成:
        1. Haplotype Token Embedding
        2. Position Embedding
        3. AF Embedding (新增!) - 使用Fourier Features编码AF

    Method: SUM (all embeddings are added together)

    Arguments:
        vocab_size: int - Vocabulary size
        embed_size: int - Embedding dimension
        dropout: float - Dropout rate
        use_af: bool - Whether to use AF embedding (default: True)
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 dropout: float = 0.1,
                 use_af: bool = True
                 ):

        super().__init__()

        # Token Embedding
        self.tokenizer = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Position Embedding
        self.position = PositionalEmbedding(embed_size)

        # AF Embedding (新增!)
        self.use_af = use_af
        if use_af:
            self.af_embedding = AFEmbedding(embed_size=embed_size, num_basis=32)

        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout)


    def forward(self, seq, af=None, pos: bool = False):
        """
        Args:
            seq: [B, L] - Token sequences
            af: [B, L] - Allele frequencies (optional, but recommended)
            pos: bool - Whether to use position embedding

        Returns:
            [B, L, embed_size] - Final embeddings
        """
        # Token embedding
        out = self.tokenizer(seq)

        # Position embedding
        if pos:
            out = out + self.position(seq)

        # AF embedding (关键改进!)
        if self.use_af and af is not None:
            af_emb = self.af_embedding(af)  # [B, L, embed_size]
            out = out + af_emb  # 加性融合，AF与token地位平等!

        return self.dropout(out)


