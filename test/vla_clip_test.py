"""
    CIS 6200 Final Project
    Unit test to test path extraction from the dataset. 
    April 2024
"""

import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.vla_dataset import VLADataset
from lib.vla_dataloader import VLADataLoader

from models.vla_clip import CLIP3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(path, model_path):

    clip = CLIP3D('train', model_path) 

    dataset = VLADataset(path)
    dataloader = VLADataLoader(dataset, 16)

    for text, image, path in dataloader:
        image = image[:,:,:,:-1]
        print("[TEST] encoding text")
        txt_output = clip.encode_text(text)
        print("[TEST] text output: ", txt_output.shape)
        print("[TEST] encoding image") 
        img_output = clip.encode_image(image)
        print("[TEST] img output: ", img_output.shape)
        print("[TEST] encoding path")
        pth_output = clip.encode_path(path)
        print("[TEST] path output: ", pth_output.shape)

    #for data in dataset:
    #    if data.text != None:
    #        print(len(data.image))
    #        print(data.text)
    #        print(data.path)


if __name__ == "__main__":

    path = "/home/jasonah/data/rxr-data/"
    model_path = "/home/jasonah/models/saved/"
    main( path, model_path )

