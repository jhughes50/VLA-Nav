"""
    CIS 6200 -- Deep Learning Final Project
    Larger transformer based path encoder
    April 2024
"""

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.positional_encoder import PositionalEncoding


class PathEncoderTransformer(nn.Module):

    def __init__(self, input_dim, output_dim, model_dim, num_heads, hidden_dim, num_layers, dropout):

        super().__init__()
       
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(model_dim, dropout, 1024)
        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        self.linear_in = nn.Linear(input_dim, model_dim)

        self.d_model_ = model_dim
        self.linear_out = nn.Linear(model_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear_in.bias.data.zero_()
        self.linear_in.weight.data.uniform_(-initrange, initrange)

        self.linear_out.bias.data.zero_()
        self.linear_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        src = self.linear_in(x)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.linear_out(output).squeeze()
        output = output.sum(0)

        return output

