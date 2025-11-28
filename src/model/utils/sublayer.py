import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x, sublayer):
        return self.dropout(self.norm(x + sublayer(x)))

