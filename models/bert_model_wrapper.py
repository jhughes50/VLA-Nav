"""
    CIS 6200 -- Deep Learning Final Project
    A wrapper around the BERT model from hugging face
    May 2024
"""

import torch
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer
from transformers import TrainingArguments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTWrapper:

    def __init__(self, mode):
        
        self.model_ = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, return_dict=True)
        self.model_.to(DEVICE)
        self.tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
        self.optimizer_ = AdamW(self.model_.parameters(), lr=1e-5)

        self.set_mode(mode)
        print("[BERT-WRAPPER] model on cuda: ", next(self.model_.parameters()).is_cuda)

    
    def model(self, ids, mask):
        return self.model_(ids, mask)

    @property
    def optimizer(self):
        return self.optimizer_

    def embed(self, text_batch):
        return self.tokenizer_(text_batch, return_tensor='pt', padding=True, truncation=False)

    def set_mode(self, mode):
        if mode == 'train':
            self.model_.train()
        elif mode == 'eval':
            self.model_.eval()
        else:
            print("[BERT-Wrapper] Model mode must be train or eval")
            exit()

    def save(self, output_dir):
        self.model_.save_pretrained(output_dir+"bert-tuned.pth")
