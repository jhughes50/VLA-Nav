"""
    CIS 6200 -- Deep Learning Final Project
    Script to train all 3d clip model
    May 2024
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from models.vla_clip import CLIP3D

from lib.vla_dataset import VLADataset
from lib.vla_dataloader import VLADataLoader

from utils.similarity import VLASimilarity
from utils.loss_logger import LossLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(clip, dataloader, optimizer, batch_size, model_path):

    similarity = VLASimilarity()

    criterion = nn.CrossEntropyLoss()

    #ti_logger = LossLogger(model_path+"clip_logs/", 'text-image')
    #tp_logger = LossLogger(model_path+"clip_logs/", 'text-path')
    #ip_logger = LossLogger(model_path+"clip_logs/", 'image-path')
    total_logger = LossLogger(model_path+"clip_logs/", 'total-corr')

    counter = 0
    save_idx = 0
    save = 1000 // batch_size

    for text, image, path, labels in dataloader:
        #print(text)
        txt_encoded = clip.encode_text(text)
        img_encoded = clip.encode_image(image)
        pth_encoded = clip.encode_path(path)

        labels = labels.to(DEVICE)
        ip_labels = torch.arange(batch_size).to(DEVICE)
            
        ti = (torch.mm(txt_encoded, img_encoded.T) + 1) / 2
        ip = F.softmax(torch.mm(img_encoded, pth_encoded.T), dim=1)
        tp = F.softmax(torch.mm(txt_encoded, pth_encoded.T), dim=1)

        print(ti.shape)
        print(ti)
        logits = (ti + ip + tp) / 3

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if counter == save:
            print("[CLIP-TRAINING] Saving models at index %s"%save_idx)
            clip.save(model_path+"clip_models/", str(save_idx))
            save_idx += 1
            counter = 0
        else:
            counter += 1
        print("[VLA-TRAIN] Loss: ", loss)
        #ti_logger.log(loss_ti)
        #tp_logger.log(loss_tp)
        #ip_logger.log(loss_ip)
        total_logger.log(loss)


if __name__ == "__main__":

    data_path = "/home/jasonah/data/rxr-data/"
    model_path = "/home/jasonah/models/saved/" 

    batch_size = 16

    dataset = VLADataset(data_path)
    dataloader = VLADataLoader(dataset, batch_size = batch_size)

    clip = CLIP3D('train', model_path)

    optimizer = torch.optim.AdamW(clip.get_params(), lr=0.1)

    train(clip, dataloader, optimizer, batch_size, model_path)
