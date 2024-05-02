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

    def __init__(self, mode, input_dim, output_dim):
        
        self.model_ = ViTModel.from_pretrained('google/vit-base-patch16-224', output_hidden_states=True)
        self.processor_ = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')
        self.model_.to(DEVICE)
        self.optimizer_ = AdamW(self.model_.parameters(), lr=2e-4)
    
        self.down_sample_ = ViTDownSample(input_dim, output_dim)
        self.down_sample_.to(DEVICE)

        self.set_mode(mode)

        print("[VIT-WRAPPER] model on cuda: ", next(self.model_.parameters()).is_cuda)
        print("[VIT-WRAPPER] downsampler on cuda: ", next(self.down_sample_.parameters()).is_cuda)

    def model(self, inputs):
        output = self.model_(**inputs)
        return self.down_sample_(output.pooler_output)

    @property
    def optimizer(self):
        return self.optimizer_

    def process(self, img_batch):
        return self.processor_(img_batch, return_tensors='pt')

    def set_mode(self, mode):
        if mode == 'train':
            self.model_.train()
            self.down_sample_.train()
        elif mode == 'eval':
            self.model_.eval()
            self.down_sample_.eval()
        else:
            print("[ViT-WRAPPER] mode %s not supported, must be train or eval")
            exit()

    def save(self, output_dir):
        self.model_.save_pretrained(output_dir+"vit-tuned.pth")


class ViTDownSample(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, pooled):
        return self.linear(pooled)
