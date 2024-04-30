"""
    CIS 6200 -- Deep Learning Final Project
    Object to organize input data
    April 2024
"""

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, ViTModel
from PIL import Image
import torch

class TextImagePathDataset(Dataset):

    def __init__(self, image_paths, text_data, pose_data=None):

        self.image_paths = image_paths
        self.text_data = text_data
        # self.poses_ = pose_data
        self.img_tokenizer = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.text_toeknizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.img_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
       
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        #process text
        text = self.text_data[idx]
        text_inputs = self.text_toeknizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_model(**text_inputs)
            last_hidden_states = outputs.last_hidden_state
            text_embedding = last_hidden_states[:, 0, :]

        #process image 
        image_path = self.image_paths[idx]
        image = Image.open(image_path) 
        img_inputs = self.img_tokenizer(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.img_model(**img_inputs)
            last_hidden_states = outputs.last_hidden_state
            img_embedding = last_hidden_states[:, 0, :]

        return text_embedding, img_embedding
