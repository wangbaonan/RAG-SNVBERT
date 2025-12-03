import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np

class Attention(nn.Module):
    """
    Compute Scaled Dot Product Attention
    """

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value:torch.Tensor,
                mask = None,
                dropout = None
                ):
        
        score = torch.matmul(query, key.mT) / math.sqrt(query.size(dim=-1))
        
        if mask is not None:
            score = score.masked_fill_(mask == 0, -1e9)

        score = F.softmax(score, dim=-1)

        if dropout is not None:
            score = dropout(score)
        
        return torch.matmul(score, value), score