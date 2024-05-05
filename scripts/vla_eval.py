"""
    CIS 6200 -- Deep Learning Final Project
    Evaluation script
    May 2024
"""

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from models.vla_clip import CLIP3D

from lib.vla_dataset import VLADataset
from lib.vla_dataloader import VLADataLoader

from utils.similarity import VLASimilarity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(clip, dataloader):

    similarity = VLASimilarity()

    for text, image, path, labels in dataloader:
        pass

if __name__ == "__main__":

    data_path = "/home/jasonah/data/rxr-data/"
    model_path = "/home/jasonah/models/saved/" 

    dataset = VLADataset(data_path, mode='eval')

    txt_path = model_path + "."
    img_path = model_path + "."
    pth_path = model_path + "."

    clip = CLIP3D('eval', model_path, img_path, txt_path, pth_path)
