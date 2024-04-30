import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Contrastive Langugae-Image-Path Pretraining
class CLIPP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CLIPP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))#learned temperature parameter

    def forward(self, text_embed, image_embed, path_embed):

        #convert to output dimension 
        text_features = self.fc1(text_embed)
        image_features = self.fc2(image_embed)
        
        #normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        path_features = F.normalize(path_features, p=2, dim=1)

        #cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_path_images = path_features @ image_features.t() * logit_scale
        logits_per_images_path = logits_per_path_images.t()
        logits_per_path_text = path_features @ text_features.t() * logit_scale
        logits_per_text_path = logits_per_path_text.t()

        return logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images
    
    def compute_loss(logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images):
        #compute loss
        loss = F.cross_entropy(logits_per_text_path, torch.arange(logits_per_text_path.shape[0]))
        loss += F.cross_entropy(logits_per_path_text, torch.arange(logits_per_path_text.shape[0]))
        loss += F.cross_entropy(logits_per_images_path, torch.arange(logits_per_images_path.shape[0]))
        loss += F.cross_entropy(logits_per_path_images, torch.arange(logits_per_path_images.shape[0]))
        return loss
        