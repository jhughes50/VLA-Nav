"""
    CIS 6200 -- Deep Learning Final Project
    Larger transformer based path encoder
    April 2024
"""

import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayers
from positonal_encoder import PositionalEncoding


class PathEncoderTransformer(nn.Module):

    def __init__(self, output_dim, model_dim, num_heads, hidden_dim, num_layers, dropout):

        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model_ = model_dim
        self.linear = nn.Linear(model_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model_)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        output = self.linear(output)

        return output

