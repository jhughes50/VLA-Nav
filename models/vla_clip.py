"""
    CIS 6200 -- Deep Learning Final Project
    Wrapper for path encoder model
    May 2024
"""

import torch
from models.bert_model_wrapper import BERTWrapper
from models.vit_model_wrapper import ViTWrapper
from models.path_model_wrapper import PathModelWrapper


class CLIP3D:

    def __init__(self, mode, input_path):

        self.img_model_ = ViTWrapper(mode)
        self.txt_model_ = BERTWrapper(mode)
        self.pth_model_ = PathModelWrapper(mode, input_path)

    def encode_text(self, text):
        emb = self.txt_model_.embed(text)
        outputs = self.txt_model_.model(emb['input_ids'], emb['attention_mask'])
        return outputs

    def encode_image(self, img):
        emb = self.img_model_.process(img)
        outputs = self.img_model_.model(emb)
        return outputs

    def encode_path(self, path):
        outputs = self.pth_model_.model(path)
        return outputs
