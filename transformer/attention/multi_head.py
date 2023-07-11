import torch
import torch.nn as nn

from .single import Attention
from ..utils import clones

class MultiHeadAttention(nn.Module):

    """
        Calculates multi-head attention of Q, K, and V where d_k = d_v = d_model / h
        h = number of heads
    """

    def __init__(self, h:int, d_model:int, dropout:float=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.dropout = nn.Dropout(p=dropout)

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, 
                query:torch.Tensor,
                key:torch.Tensor, 
                value:torch.Tensor, 
                mask:torch.Tensor=None):
        
        if mask is not None:
            # mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.shape[0]

        # linear projections of Q, K, and V
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # apply attention to projected vectors
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # concat with view
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)