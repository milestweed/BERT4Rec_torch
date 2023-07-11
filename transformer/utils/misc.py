import copy

import torch
import torch.nn as nn


def clones(module, N: int) -> nn.ModuleList:
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size: int) -> torch.Tensor:
    "Mask out subsequent position"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0
