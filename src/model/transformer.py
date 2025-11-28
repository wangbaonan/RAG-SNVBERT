import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .utils import SublayerConnection, FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHeadAttention + FeedForward with sublayer connection
    """

    def __init__(self, dims, attn_heads, feed_forward_hidden, dropout):
        """
        Arguments:

            dims: dimension of the input and output
            attn_heads: head size of multi-head attention
            feed_forward_hiddhen: usually 4 * dims
            dropout: dropout rate
        """

        super().__init__()

        self.attention = MultiHeadAttention(heads=attn_heads, dims=dims)
        self.feed_forward = FeedForward(dims=dims, hidden_dims=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=dims, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=dims, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, mask = None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
        