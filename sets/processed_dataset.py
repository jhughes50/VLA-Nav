"""
    CIS 6200 -- Deep Learning Final Project
    Creates a torch Dataset that uses the preprocessed images and paths
    May 2024
"""

import json
import gzip

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VLAProcessedDataset(Dataset):

    def __init__(self, file_path, mode='train'):
        super().__init__()
        
        self.rxr_guide_ = list()
        self.file_guide_ = list()

        self.file_path_ = file_path

        self.load_guide(file_path, mode)
        self.load_file_guide(file_path, mode)

        self.transform_ = transforms.Compose([ transforms.PILToTensor() ])

    def load_guide(self, path, mode):
        if mode == 'train':
            path = path + "rxr_train_guide.jsonl.gz"
        elif mode == 'eval':
            path = path + "rxr_val_seen_guide.jsonl.gz"
        else:
            print("[VLA-DATASET] Unrecognized mode: %s, use train or eval" %mode)

        with gzip.open(path, 'r') as f:
            print("[VLA-DATASET] found file %s , loading..." %path)
            self.rxr_guide_ = [json.loads(line) for line in f]
            print("[VLA-DATASET] guide loaded")


    def load_file_guide(self,path, mode):
        if mode == 'train':
            path = path + "train/rxr_train_file_guide.txt"
        elif mode == 'eval':
            path = path + "eval/rxr_eval_senn_file_guide.txt"
        else:
            print("[VLA-DATASET] Unrecognized mode: %s, use train or eval" %mode)
        
        with open(path, 'r') as f:
            print("[VLA-DATASET] looking for file at: %s, loading..." %path)
            self.file_guide_ = f.readlines()
        
        if len(self.file_guide_) == 0:
            raise FileNotFoundError("Could not find file at %s" %path)


    def __len__(self):
        return len(self.file_guide_)//2

    def get_image(self, file_name):
        path = self.file_path_ + "train/" + file_name
        img = Image.open(path)
     
        return self.transform_(img)

    def get_path(self, file_name):
        path = self.file_path_ + "train/" + file_name
        agent_path = np.loadtxt(path)
   
        return torch.Tensor(agent_path)

    def get_text(self, iid):
        for desc in self.rxr_guide_:
            desc_id = desc['instruction_id']
            if "{:06}".format(desc_id) == iid:
                return desc['instruction']

    def __getitem__(self, idx):
        pth_file = self.file_guide_[2*idx].split('\n')[0]
        jpg_file = self.file_guide_[2*idx+1].split('\n')[0]
        print(pth_file) 
        iid = pth_file.split('_')[0]
        
        agent_path = self.get_path(pth_file)
        agent_img  = self.get_image(jpg_file)
        agent_txt  = self.get_text(iid)
        
        return agent_txt, agent_img, agent_path

