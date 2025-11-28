import torch.nn as nn
import torch

import math

MAX_SEQ_LEN = 1030


class PositionalEmbedding(nn.Module):
    """
    Arguments:

        dims: int, dimension of the embedding layer
        max_len: int, max length of the sentence
    """

    def __init__(self,
                 dims: int,
                 max_len: int = MAX_SEQ_LEN
                 ):
        
        super().__init__()

        pe = torch.zeros([max_len, dims]).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, dims, 2).float() * -(math.log(10000.0) / dims)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        return self.pe[:, :x.size(dim=1)]

