import torch
import torch.nn as nn

from typing import Optional

# from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding is a combination consisted of:
        
        1. Haplotype Token Embedding.
        3. Position Embedding.

    Method: SUM.

    Arguments:

        vocab_size: int
        embed_size: int
        dropout: float
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 dropout: float = 0.1
                 ):
        
        super().__init__()

        # Vocabulary Embedding
        self.tokenizer = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.position = PositionalEmbedding(embed_size)

        self.embed_size = embed_size


    def forward(self, seq, pos : bool = False):
        out = self.tokenizer(seq)

        if pos:
            out += self.position(seq)

        return out


