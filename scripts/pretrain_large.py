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
from models.path_encoder_large import PathEncoderTransformer
from models.path_decoder_large import PathDecoderTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = "/home/jasonah/data/"
MODEL_PATH = "/home/jasonah/models/saved/"

DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(encoder, decoder, optimizer, dataloader, epochs):
    
    encoder.train()
    decoder.train()

    encoder.to(DEVICE)
    decoder.to(DEVICE)

    print("[PRETRAIN] Encoder on cuda: ", next(encoder.parameters()).is_cuda)
    print("[PRETRAIN] Decoder on cuda: ", next(decoder.parameters()).is_cuda)

    torch.save(encoder.state_dict(), MODEL_PATH+"encoder_test.pth")
    ft = open(MODEL_PATH+"loss_total.log", 'w')
    
    criterion = nn.MSELoss()

    for ep in range(epochs):
        print("[PRETRAIN] Training epoch %s..." %ep)
        fe = open(MODEL_PATH+"loss_epoch_%s.log" %ep, 'w')
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
        
            fe.write(str(loss.item())+"\n")
        print("[PRETRAIN] Saving epoch: %s with loss: %s" %(ep, loss))
        # save the model as each epoch
        fe.close()
    
        torch.save(encoder.state_dict(), MODEL_PATH+"encoder_epoch%s.pth" %ep)
        torch.save(decoder.state_dict(), MODEL_PATH+"decoder_epoch%s.pth" %ep)
    
        ft.write(loss.item()+"\n")
    
    ft.close()

    print("[PRETRAIN] Finished")

if __name__ == "__main__":
   
    input_dim = 96
    hidden_dim = 256
    output_dim = 512
    model_dim = 256
    num_heads = 2
    num_layers = 2
    dropout = 0.2

    epochs = 10

    dataloader = PathDataLoader(DATA_PATH, interpolate=True, out_dim=96)

    encoder = PathEncoderTransformer(input_dim, output_dim, model_dim, num_heads, hidden_dim, num_layers, dropout)
    decoder = PathDecoderTransformer(input_dim, output_dim, model_dim, num_heads, hidden_dim, num_layers, dropout)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    print("[PRETRAIN] Starting pretraining")
    train(encoder, decoder, optimizer, dataloader, epochs)
    print("[PRETRAIN] Training finished")
