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

    def __init__(self, mode, input_dim, output_dim, model_path):
        
        if model_path == None:
            self.model_ = ViTModel.from_pretrained('google/vit-base-patch16-224', output_hidden_states=True)
        else:
            self.model_ = ViTModel.from_pretrained(model_path, output_hidden_states=True)

        self.processor_ = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')
        self.model_.to(DEVICE)
    
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

    def get_params(self):
        return list(self.model_.parameters()) + list(self.down_sample_.parameters())

    def save(self, output_dir, idx):
        print("[VIT-WRAPPER] Saving model to %s at index %s" %(output_dir, idx))
        self.model_.save_pretrained(output_dir+"vit-tuned-%s"%idx, from_pt=True)
        torch.save(self.down_sample_.state_dict(), output_dir+"vit-linear-%s.pth"%idx)


class ViTDownSample(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, pooled):
        return self.linear(pooled)
