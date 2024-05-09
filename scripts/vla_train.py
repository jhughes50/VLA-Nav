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

    ti_logger = LossLogger(model_path+"clip_logs/", 'text-image-2')
    tp_logger = LossLogger(model_path+"clip_logs/", 'text-path-2')
    ip_logger = LossLogger(model_path+"clip_logs/", 'image-path-2')
    total_logger = LossLogger(model_path+"clip_logs/", 'total-2')

    counter = 0
    save_idx = 8
    save = 1000 // batch_size

    for text, image, path, labels in dataloader:
        #print(text)
        txt_encoded = clip.encode_text(text)
        img_encoded = clip.encode_image(image)
        pth_encoded = clip.encode_path(path)

        sim_it = similarity.get_logits(img_encoded, txt_encoded) * clip.temp
        sim_tp = similarity.get_logits(txt_encoded, pth_encoded) * clip.temp
        sim_ip = similarity.get_logits(img_encoded, pth_encoded) * clip.temp

        print(sim_it)

        labels = labels.to(DEVICE)
        ip_labels = torch.arange(batch_size).to(DEVICE)

        #ti_target = similarity.get_target(img_encoded, txt_encoded).to(DEVICE)
        #tp_target = similarity.get_target(txt_encoded, pth_encoded).to(DEVICE)
        #ip_target = similarity.get_target(img_encoded, pth_encoded).to(DEVICE)

        #loss_ti = (criterion(sim_it, ti_target) + criterion(sim_it.T, ti_target.T)) / 2
        #loss_tp = (criterion(sim_tp, tp_target) + criterion(sim_tp.T, tp_target.T)) / 2
        #loss_ip = (criterion(sim_ip, ip_target) + criterion(sim_ip.T, ip_target.T)) / 2
        
        loss_ti = (criterion(sim_it, labels) + criterion(sim_it.T, ip_labels) ) / 2
        loss_tp = (criterion(sim_tp, labels) + criterion(sim_tp.T, ip_labels) ) / 2
        loss_ip = (criterion(sim_ip, ip_labels) + criterion(sim_ip.T, ip_labels) ) / 2

        loss = (loss_ti + loss_tp + loss_ip) / 3

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
        ti_logger.log(loss_ti)
        tp_logger.log(loss_tp)
        ip_logger.log(loss_ip)
        total_logger.log(loss)


if __name__ == "__main__":

    data_path = "/home/jasonah/data/rxr-data/"
    model_path = "/home/jasonah/models/saved/" 

    batch_size = 16

    dataset = VLADataset(data_path)
    dataloader = VLADataLoader(dataset, batch_size = batch_size)

    clip = CLIP3D('train', model_path)

    optimizer = torch.optim.SGD(clip.get_params(), lr=1e-5)

    train(clip, dataloader, optimizer, batch_size, model_path)
