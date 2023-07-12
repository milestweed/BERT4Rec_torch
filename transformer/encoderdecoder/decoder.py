import torch
import torch.nn as nn

from ..utils import LayerNorm, SublayerConnection, clones

class Decoder(nn.Module):
    "Generic N layer decoder with masking"

    def __init__(self, layer:nn.Module, N:int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, 
                x:torch.Tensor, 
                memory:torch.Tensor, 
                src_mask:torch.Tensor, 
                tgt_mask:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask):
        return self.norm(x)
    


class DecoderLayer(nn.Module):
    "The decoder is made of self-attn, src-attn, and feed forward"

    def __init__(self, 
                    size:int, 
                    self_attn:nn.Module, 
                    src_attn:nn.Module, 
                    feed_forward:nn.Module, 
                    dropout:float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, 
                x:torch.Tensor, 
                memory:torch.Tensor, 
                src_mask:torch.Tensor, 
                tgt_mask:torch.Tensor) -> nn.Module:
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)