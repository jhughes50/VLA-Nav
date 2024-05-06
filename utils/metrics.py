"""
    CIS 6200 -- Deep Learning Final Project
    class of metrics to calculate
    May 2024
"""

import torch
import torch.nn.functional as F

class VLAMetrics:

    def mse(self, inp, trg):
        return F.mse_loss(inp, trg)

    def destination(self, inp, trg):
        pred = inp[:,-1]
        label = trg[:,-1]

        return torch.norm(pred-label)

    def mae(self, inp, trg):
        return F.mae_loss(inp, trg)

    def success(self, inp, trg, threshold = 5):
        n = self.destination(inp, trg)
        if n < threshold:
            return 1
        else:
            return 0

    def classification(self, similar, target):
        if similar == target:
            return 1
        else:
            return 0
