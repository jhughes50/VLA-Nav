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
       
        self.pos_encoder = nn.Parameter(torch.randn(input_dim, 1, model_dim))
        self.input_projection = nn.Sequential(nn.Conv1d(input_dim, 128, 1), nn.ReLU(),
                                              nn.Conv1d(128, 256, 1), nn.ReLU(),
                                              nn.Conv1d(256, model_dim, 1), nn.ReLU())


        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.d_model_ = model_dim
        self.linear_out = nn.Linear(model_dim, output_dim)


    def forward(self, x):
        src = self.input_projection(x).permute(2,0,1)

        output = self.transformer_encoder(src)
        output = self.linear_out(output).squeeze()
        output = output.mean(0)

        return output

