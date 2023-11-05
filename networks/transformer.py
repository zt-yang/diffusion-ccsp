import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.set_printoptions(edgeitems=16)
from collections import OrderedDict
import math
import numpy as np
from einops import rearrange, reduce, repeat
import copy
import random


class PositionalEncoding(nn.Module):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.dropout = nn.Dropout(p=0.1)
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor=None):
        # if attn_mask is not None:
        #     attn_mask = attn_mask[:x.shape[0], :x.shape[0]].to(x.device, non_blocking=True)
        return self.attn(x, x, x, attn_mask=attn_mask)

    def forward(self, x: tuple):
        x, weights, mask = x
        attn, attn_weights = self.attention(self.ln_1(x), mask)
        if weights is None:
            weights = attn_weights.unsqueeze(1)
        else:
            weights = torch.cat([weights, attn_weights.unsqueeze(1)], dim=1)
        x = x + attn
        x = x + self.ln_2(self.dropout(self.mlp(x)))
        return x, weights, mask


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
