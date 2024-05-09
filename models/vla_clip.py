"""
    CIS 6200 -- Deep Learning Final Project
    Wrapper for path encoder model
    May 2024
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.bert_model_wrapper import BERTWrapper
from models.vit_model_wrapper import ViTWrapper
from models.path_model_wrapper import PathModelWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIP3D:

    def __init__(self,
                 mode,
                 input_path,
                 input_dim=768,
                 output_dim=512,
                 img_model_path = None,
                 txt_model_path = None,
                 pth_model_path = None):

        print("[CLIP-3D] Getting and setting models to %s mode" %mode)
        self.mode_ = mode
        self.temp_ = Temperature()
        self.img_model_ = ViTWrapper(mode, input_dim, output_dim, img_model_path)
        self.txt_model_ = BERTWrapper(mode, input_dim, output_dim, txt_model_path)
        self.pth_model_ = PathModelWrapper(mode, input_path, pth_model_path)

    @property
    def temp(self):
        return self.temp_.temp

    def encode_text(self, text):
        emb = self.txt_model_.embed(text)
        outputs = self.txt_model_.model(emb.to(DEVICE))
        return F.normalize(outputs)

    def encode_image(self, img):
        emb = self.img_model_.process(img)
        outputs = self.img_model_.model(emb.to(DEVICE))
        return F.normalize(outputs)

    def encode_path(self, path):
        outputs = self.pth_model_.model(path.to(DEVICE))
        if self.mode_ == 'train':
            return F.normalize(outputs, dim=1)
        else:
            outputs = outputs.unsqueeze(0)
            return F.normalize(outputs, dim=1)

    def get_params(self):
        img_params = self.img_model_.get_params()
        txt_params = self.txt_model_.get_params()
        pth_params = self.pth_model_.get_params()
        tmp_params = list(self.temp_.parameters())

        return img_params + txt_params + pth_params + tmp_params

    def save(self, output_dir, idx):
        self.txt_model_.save(output_dir, idx)
        self.img_model_.save(output_dir, idx)
        self.pth_model_.save(output_dir, idx)

class Temperature(nn.Module):

    def __init__(self):
        super().__init__()
        self.p_ = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    @property
    def temp(self):
        return torch.exp(self.p_)
