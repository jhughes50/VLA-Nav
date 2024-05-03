"""
    CIS 6200 -- Deep Learning Final Project
    scripts for calulating the similarity
    May 2024
"""

import torch
import torch.nn.functional as F

class VLASimilarity:

    def get_cosine_similarity(self, vec1, vec2):
        norm1 = torch.norm(vec1, dim=1)
        norm2 = torch.norm(vec2, dim=1)
        return torch.sum(vec1 * vec2, dim=1) / (norm1*norm2)

    def get_logits(self, vec1, vec2):
        return torch.tensordot(vec1, vec2.T, dims=1)

    def get_target(self, img, txt, pth):
        img_target = img @ img.T
        txt_target = txt @ txt.T
        pth_target = pth @ pth.T

        return F.softmax((img_target + txt_target + pth_target)/3, dim=-1)
