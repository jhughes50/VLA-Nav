"""
    CIS 6200 Final Project 
    Dataloader for path autoencoder
    April 2024
"""

import torch
from torch.utils.data import Dataset
from lib.pose_extractor import PoseExtractor

class PathDataLoader(Dataset, PoseExtractor):

    def __init__(self, path, interpolate=True, out_dim=96):
        # Inherit the pose extractor
        # @param: path: the input path to the training or val guide
        # @param: interpolate: whether or not to make interpolate the output path
        # @param: out_dim: the dimension of the output path when interpolated
        super().__init__(path, out_dim)
        self.interp_ = interpolate
        self.load()

    def __len__(self):
        return len(self.train_guide_)

    def __getitem__(self, idx):
        if self.interp_:
            path = self.path_from_guide(idx)
            path = torch.tensor(self.interpolator_.interpolate(path))
            return path.to(torch.float)
        else:
            path = torch.tensor(self.path_from_guide(idx))
            return path.to(torch.float)
