import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    "Implement the PE function"

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arrange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arrange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
