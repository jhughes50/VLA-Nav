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
from models.path_decoder_medium import PathDecoderMedium

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

DATA_PATH = "/home/jasonah/data/"
MODEL_PATH = "/home/jasonah/models/saved/"

DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def train(encoder, decoder, optimizer, dataloader, epochs):
    
    encoder.train()
    decoder.train()

    encoder.to(DEVICE)
    decoder.to(DEVICE)

    print("[PRETRAIN] Encoder on cuda: ", next(encoder.parameters()).is_cuda)
    print("[PRETRAIN] Decoder on cuda: ", next(decoder.parameters()).is_cuda)

    torch.save(encoder.state_dict(), MODEL_PATH+"encoder_test.pth")
    ft = open(MODEL_PATH+"loss__large_total.log", 'w')
    
    criterion = nn.MSELoss()

    for ep in range(epochs):
        print("[PRETRAIN] Training epoch %s..." %ep)
        fe = open(MODEL_PATH+"loss__large_epoch_%s.log" %ep, 'w')
        counter = 1
        avg_loss = 0
        total_loss = 0
        for p in dataloader:
            if isinstance(p, type(None)):
                continue
            if p.shape[0] != BATCH_SIZE:
                continue 

            p = p.reshape((BATCH_SIZE,32,3))
    
            p = p.to(DEVICE)
            encoded = encoder(p)
            decoded = decoder(encoded)

            decoded = decoded.reshape((BATCH_SIZE,32,3))
            loss = criterion(decoded, p)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 10)
            optimizer.step()
        
            total_loss += loss.item()
            avg_loss = total_loss/counter
            
            counter += 1
 
            fe.write(str(avg_loss)+"\n")

            if torch.sum(torch.isnan(decoded)) > 0:
                print("[PRETRAIN] NaN in logits")
                exit()

        print("[PRETRAIN] Saving epoch: %s with loss: %s" %(ep, avg_loss))

        fe.close()
        # save the model epoch 
        torch.save(encoder.state_dict(), MODEL_PATH+"encoder_large_epoch%s.pth" %ep)
        torch.save(decoder.state_dict(), MODEL_PATH+"decoder_large_epoch%s.pth" %ep)
    
        ft.write(str(avg_loss)+"\n")
    
    ft.close()

    print("[PRETRAIN] Finished")

if __name__ == "__main__":
   
    input_dim = 96
    input_dim_1 = 32
    hidden_dim = 256
    output_dim = 512
    model_dim = 512
    num_heads = 4
    num_layers = 6
    dropout = 0.2

    epochs = 10

    dataset = PathDataLoader(DATA_PATH, interpolate=True, out_dim=96)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    encoder = PathEncoderTransformer(input_dim_1, output_dim, model_dim, num_heads, hidden_dim, num_layers, dropout)
    decoder = PathDecoderMedium(input_dim, hidden_dim, output_dim) 
    #decoder = PathDecoderTransformer(input_dim, output_dim, model_dim, num_heads, hidden_dim, num_layers, dropout)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    print("[PRETRAIN] Starting pretraining")
    train(encoder, decoder, optimizer, dataloader, epochs)
    print("[PRETRAIN] Training finished")
