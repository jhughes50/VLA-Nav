"""
    CIS 6200 -- Deep Learning Final Project
    dataloader for vision, language and path
    April 2024
"""
import torch

class VLADataLoader:

    def __init__(self, dataset, batch_size=16):

        self.dataset_ = dataset
        self.batch_size_ = batch_size
        self.slice_ = [0, batch_size]

    def __len__(self):
        return len(self.dataset_.be.guide)// self.batch_size_

    def __getitem__(self, idx):
        self.slice_ = [idx*self.batch_size_, idx*self.batch_size_ + self.batch_size_]
        self.batchify()

    def batchify(self, idx):
        # loop and create combinations
        fill = 0
        while fill < self.batch_size_:
            pass
