import torch
import torch.nn as nn

from .single import Attention

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

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_layer = nn.Linear(d_model, d_model)
        self.attn = Attention()

    def forward(self, 
                query:torch.Tensor, 
                key:torch.Tensor, 
                value:torch.Tensor, 
                mask:torch.Tensor=None):
        
        # one batch for each user
        nbatches = query.shape[0]