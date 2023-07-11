import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
        Calculate scaled dot product attention with [optional] masking and dropout
    """
    def forward(self, 
                query:torch.Tensor, 
                key:torch.Tensor, 
                value:torch.Tensor, 
                mask:torch.Tensor=None, 
                dropout:nn.Module=None) -> tuple(torch.Tensor, torch.Tensor):
        
        d_k = query.shape[-1]

        scores = torch.matmul(query, key.transpose(-2,-1)) \
                / math.sqrt(d_k)
        
        if mask is not None:
            scores.masked_fill(mask == 0, -1e-9)
        
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn