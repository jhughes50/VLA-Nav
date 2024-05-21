"""
    CIS 6200 -- Deep Learning Final Project
    Training script for processed data
    May 2024
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader

from sets.processed_dataset import VLAProcessedDataset
from models.vla_clip import CLIP3D

from utils.similarity import VLASimilarity
from utils.loss_logger import LossLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODE = 'train'

def train(clip, dataloader, optimizer):
    
    for text, image, path in dataloader:
        print(type(text))
        print(image.shape)
        print(path.shape)
        break


if __name__ == "__main__":

    data_path = "/home/jasonah/data/rxr-data/"
    model_path = "/home/jasonah/models/saved/" 

    batch_size = 16

    dataset = VLAProcessedDataset(data_path, mode=MODE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    clip = None #CLIP3D(MODE, model_path)
    
    optimizer = None #torch.optim.Adam(clip.get_params(), lr=1e-7)

    train( clip, dataloader, optimizer )


