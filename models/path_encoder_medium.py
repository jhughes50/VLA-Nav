"""
    CIS 6200 - Deep Learning
    Autoencoder model for Pose Encoding 
    April 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    NOTE:
    Notice in_dim is the dimension of our original input so it is
    the dimension we want the decoder to output, hence layer3 of the 
    decoder outputs in_dim.
"""

class PathEncoderMedium(nn.Module):
    # encode poses of path, by compressing them to out_dim
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        # layers
        self.layer1_ = nn.Linear(in_dim, 256)
        self.layer2_ = nn.Linear(256, 512)
        self.layer3_ = nn.Linear(512, 768)
        self.layer4_ = nn.Linear(768, 768)
        self.layer5_ = nn.Linear(768, out_dim)

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
