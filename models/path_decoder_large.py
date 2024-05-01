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
from models.positional_encoder import PositionalEncoding


class PathDecoderTransformer(nn.Module):

    def __init__(self, output_dim, input_dim, model_dim, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()

        self.model_type = 'Transformer' 
        self.pos_encoder = PositionalEncoding(model_dim, dropout, 1024)
        decoder_layer = TransformerEncoderLayer(model_dim, num_heads)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers)
        
        self.d_model_ = model_dim
        self.linear_in = nn.Linear(input_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, output_dim)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1

        self.linear_in.bias.data.zero_()
        self.linear_in.weight.data.uniform_(-initrange, initrange)

        self.linear_out.bias.data.zero_()
        self.linear_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, emb):
        emb = self.linear_in(emb)
        #emb = self.pos_encoder(emb)
        output = self.transformer_decoder(emb)
        
        output = self.linear_out(output).squeeze()
        logits = output.sum(0)
        
        return logits

