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

    def get_1d_logits(self, v1, v2):
        return torch.dot(v1,v2)

    def get_blocked_logits(self, vec1, vec2):
        dotted = torch.tensordot(vec1, vec2.T, dims=1)

    def get_target(self, v1, v2):
        v1_target = torch.tensordot(v1, v1.T, dims=1)
        v2_target = torch.tensordot(v2, v2.T, dims=1) 
        #return (v1+v2)/2
        return F.softmax((v1_target + v2_target)/2, dim=-1)

    def get_correlation_logits(self, img, txt, pth):
        # this is wrong
        X = torch.stack((img,txt,pth), dim=0)
        corr = torch.bmm(X,X.mT)
        U, S, Vt = torch.linalg.svd(corr)
        print(S)
        return torch.sum(S, dim=0)/3
