import torch.nn as nn
import torch
from .attention import Attention


class MultiHeadAttention(nn.Module):
    """
    Arguments:
    
        heads: int
        dims: dimension of the hidden layer
        dropout: float
    """

    def __init__(self,
                 heads: int,
                 dims: int,
                 dropout: float = 0.1
                 ):
        
        super().__init__()
        
        assert dims % heads == 0, "Hidden dimension must be divisible by Heads"

        self.heads = heads
        self.dims = dims // heads

        self.linear_layers = nn.ModuleList([nn.Linear(dims, dims) for _ in range(3)])
        self.output_layer = nn.Linear(dims, dims)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask = None
                ):
        
        batch_size = query.size(dim=0)

        query, key, value = [layer(x).view(batch_size, -1, self.heads, self.dims).transpose(1, 2)
                             for layer, x in zip(self.linear_layers, (query, key, value))]

        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dims)

        return self.output_layer(x)

        
