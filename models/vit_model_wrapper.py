"""
    CIS 6100 -- Deep Learning Final Project
    Wrapper for the ViT model for fine-tuning
    May 2024
"""
import torch
from torch import nn
from transformers import ViTForImageClassification
from transformers import ViTModel
from transformers import ViTImageProcessor
from transformers import AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTWrapper:

    def __init__(self, mode):
        
        self.model_ = ViTModel.from_pretrained('google/vit-base-patch16-224', output_hidden_states=True)
        self.processor_ = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')
        self.model_.to(DEVICE)
        self.optimizer_ = AdamW(self.model_.parameters(), lr=2e-4)
        self.set_mode(mode)

        print("[VIT-WRAPPER] model on cuda: ", next(self.model_.parameters()).is_cuda)

    def model(self, inputs):
        output = self.model_(**inputs)
        return output.pooler_output

    @property
    def optimizer(self):
        return self.optimizer_

    def process(self, img_batch):
        return self.processor_(img_batch, return_tensors='pt')

    def set_mode(self, mode):
        if mode == 'train':
            self.model_.train()
        elif mode == 'eval':
            self.model_.eval()
        else:
            print("[ViT-WRAPPER] mode %s not supported, must be train or eval")
            exit()

    def save(self, output_dir):
        self.model_.save_pretrained(output_dir+"vit-tuned.pth")


class ImgHidden(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, pooled):
        return self.linear(pooled)
