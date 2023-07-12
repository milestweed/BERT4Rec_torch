import copy

import torch
import torch.nn as nn

from utils import PositionwiseFeedForward, SinusoidalPositionEncoding, Embeddings
from attention import MultiHeadAttention
from encoderdecoder import (
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
    EncoderDecoder,
    Generator,
)


def make_model(src_items, tgt_items, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: construct model from hyperparameters"
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = SinusoidalPositionEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_items), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_items), c(position)),
        Generator(d_model, tgt_items),
    )

    # Glorot initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
