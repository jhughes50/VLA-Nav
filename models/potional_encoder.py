"""
    CIS 6200 -- Deep Learning Final Project
    positional encoder for transformer encoder
    April 2024
"""

import math
import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len),unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))

        pe = torch.zeros(max_len, 1, model_dim)
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * dov_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]

        return self.dropout(x)
