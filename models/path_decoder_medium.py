"""
    CIS 6200 - Deep Learning
    A pose decoder as part of the autoencoder
    pretraining for embedding the action space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PathDecoderMedium(nn.Module):
    # decode poses from path, by taking them from the 
    # encoder embedding space to input space
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        # layers
        self.layer1_ = nn.Linear(out_dim, 768)
        self.layer2_ = nn.Linear(768, 768)
        self.layer3_ = nn.Linear(768, 512)
        self.layer4_ = nn.Linear(512, 256)
        self.layer5_ = nn.Linear(256, in_dim)

        self.dropout_ = nn.Dropout(0.2)

    def forward(self, x):
        # forward pass
        out = F.relu(self.layer1_(x))
        out = self.dropout_(out)
        out = F.relu(self.layer2_(out))
        out = F.relu(self.layer3_(out))
        out = self.dropout_(out)
        out = F.relu(self.layer4_(out))
        out = F.relu(self.layer5_(out))

        return out
