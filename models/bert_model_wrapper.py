"""
    CIS 6200 -- Deep Learning Final Project
    A wrapper around the BERT model from hugging face
    May 2024
"""

import torch
from torch import nn
from transformers import RobertaModel
from transformers import RobertaTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTWrapper:

    def __init__(self, mode, input_dim, output_dim):
        
        self.model_ = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        self.model_.to(DEVICE)
        
        self.tokenizer_ = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.down_sample_ = BERTDownSample(input_dim, output_dim)
        self.down_sample_.to(DEVICE)

        self.set_mode(mode)
        print("[BERT-WRAPPER] model on cuda: ", next(self.model_.parameters()).is_cuda)
        print("[BERT-WRAPPER] downsample layer on cuda: ", next(self.down_sample_.parameters()).is_cuda)
    
    def model(self, inputs):
        output = self.model_(**inputs)
        return self.down_sample_(output.pooler_output)

    @property
    def optimizer(self):
        return self.optimizer_

    def embed(self, text_batch):
        return self.tokenizer_(text_batch, return_tensors='pt', padding=True, truncation=True)

    def set_mode(self, mode):
        if mode == 'train':
            self.model_.train()
            self.down_sample_.train()
        elif mode == 'eval':
            self.model_.eval()
            self.down_sample_.train()
        else:
            print("[BERT-Wrapper] Model mode must be train or eval")
            exit()

    def get_params(self):
        return list(self.model_.parameters()) + list(self.down_sample_.parameters())

    def save(self, output_dir, idx):
        print("[BERT-WRAPPER] Saving model to %s with index %s" %(output_dir, idx))
        self.model_.save_pretrained(output_dir+"bert-tuned-%s"%idx, from_pt=True)
        torch.save(self.down_sample_.state_dict(), output_dir+"bert-linear-%s.pth"%idx)

class BERTDownSample(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, pooled):
        return self.linear(pooled)
