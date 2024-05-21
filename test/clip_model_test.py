"""
    CIS 6200 -- Deep Learning Final Project
    test script to debug the processed torch dataset
    May 2024
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from sets.processed_dataset import VLAProcessedDataset
from models.clip_model_wrapper import CLIPWrapper


if __name__ == "__main__":

    path = "/home/jasonah/data/rxr-data/"

    dataset = VLAProcessedDataset(path, mode='train')
    print(type(dataset.file_guide_))
    clip = CLIPWrapper('train', 'cuda')
    c = 0
    for p,i,t in dataset:
        print("[TEST] index %s: text %s img %s txt %s" %(c,p.shape,i.shape,t[:10]))
        te, ie = clip.model(t,i)
        print(te.shape, ie.shape)
        if c == 5:
            break
        c += 1
