"""
    CIS 6200 -- Deep Learning Final Project
    Dataloader for all vision language and path
    April 2024
"""

from preprocess.base_extractor import BaseExtractor
from torch.utils.data import Dataset

class VLADataset(Dataset):

    def __init__(self, file_path, mode='train'):
        super().__init__()
        self.be = BaseExtractor(file_path, mode)

    def __len__(self):
        return len(be.guide)

    def __getitem__(self, idx):
        return self.be.extract(idx)
