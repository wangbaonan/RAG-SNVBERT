import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, dims, hidden_dims, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(dims, hidden_dims)
        self.w_2 = nn.Linear(hidden_dims, dims)

        self.activation1 = nn.LeakyReLU(negative_slope=0.1)
        self.activation2 = nn.LeakyReLU(negative_slope=0.1)

        self.norm = nn.LayerNorm(hidden_dims)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.activation1(self.w_1(x))
        x = self.activation2(self.w_2(self.norm(x)))
        return self.dropout(x)