import torch
import torch.nn as nn

from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection fallowedby a layer norm.
    Note: for code simplicty the norm is first as opposed to last
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))
