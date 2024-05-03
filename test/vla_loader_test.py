"""
    CIS 6200 Final Project
    Unit test to test path extraction from the dataset. 
    April 2024
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.vla_dataset import VLADataset
from lib.vla_dataloader import VLADataLoader

def main(path):

    dataset = VLADataset(path)
    dataloader = VLADataLoader(dataset, 16)

    for text, image, path, labels in dataloader:
        print(len(text)) 
        print(image.shape)
        print(path.shape)
        print(labels)
        print(len(labels))

    #for data in dataset:
    #    if data.text != None:
    #        print(len(data.image))
    #        print(data.text)
    #        print(data.path)


if __name__ == "__main__":

    path = "/home/jasonah/data/rxr-data/"
    main( path )

