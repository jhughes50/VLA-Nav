"""
    CIS 6200 -- Deep Learning Final Project
    pose decoder for transformer based autoencoder
    April 2024

    NOTE: It doesn't make sense to use the TransformerDecoder methods from pytorch
    here because it requires a shifted target input (i.e. a passthrough). Of course,
    this would allow the decoder to cheat, thus we use the encoder moethods here soley to 
    bring the embedding back to the input dimensions, thus making it a decoder.
"""

import math 
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PathDecoderTransformer(nn.Module):

    def __init__(self, output_dim, model_dim, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()

        self.model_type = 'Transformer'
        decoder_layer = TransformerEncoderLayer(model_dim, num_heads)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers)
        
        self.d_model_ = model_dim
        self.linear = nn.Linear(model_dim, output_dim)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, emb):
        output = self.transformer_decoder(emb)
        output = self.linear(output)

        return output

