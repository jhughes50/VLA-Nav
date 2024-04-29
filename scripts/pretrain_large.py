"""
    CIS 6200 - Deep Learning
    Autoencoder driver software 
    Trains an autoencoder on pose data
    from habitat
    April 2024
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.path_dataloader import PathDataLoader
from lib.pose_extractor import PoseExtractor
from models.pose_encoder_large import PoseEncoderTransformer
from models.pose_decoder_large import PoseDecoderTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = "/home/jasonah/data/VLA-Nav-Data/rxr-data/rxr_train_guide.jsonl.gz"
MODEL_PATH = "/home/jasonah/models/"

DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(encoder, decoder, optimizer, dataloader, epochs):
    
    encoder.train()
    decoder.train()

    encoder.to(DEVICE)
    decoder.to(DEVICE)

    print("[PRETRAIN] Encoder on cuda: ", next(encoder.parameters()).is_cuda)
    print("[PRETRAIN] Decoder on cuda: ", next(decoder.parameters()).is_cuda)

    criterion = nn.MSELoss()

    for ep in range(epochs):
        print("[PRETRAIN] Training epoch %s..." %ep)
        for p in dataloader:
            if isinstance(p, type(None)):
                continue
            p = p.to(DEVICE)
            encoded = encoder(p)
            decoded = decoder(encoded)

            loss = criterion(decoded, p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print("[PRETRAIN] Saving epoch: %s with loss: %s" %(ep, loss))
        # save the model as each epoch
        
        torch.save(encoder.state_dict(), MODEL_PATH+"encoder_epoch%s.pth" %ep)
        torch.save(decoder.state_dict(), MODEL_PATH+"decoder_epoch%s.pth" %ep)
    

if __name__ == "__main__":
   
    input_dim = 96
    hidden_dim = 512
    output_dim = 512

    epochs = 10

    dataloader = PathDataLoader(DATA_PATH, interpolate=True, out_dim=96)

    encoder = PoseEncoder(input_dim, hidden_dim, output_dim)
    decoder = PoseDecoder(input_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    print("[PRETRAIN] Starting pretraining")
    train(encoder, decoder, optimizer, dataloader, epochs)
    print("[PRETRAIN] Training finished")
