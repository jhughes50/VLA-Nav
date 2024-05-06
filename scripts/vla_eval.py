"""
    CIS 6200 -- Deep Learning Final Project
    Evaluation script
    May 2024
"""
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from models.vla_clip import CLIP3D

from lib.vla_dataset import VLADataset
from lib.vla_dataloader import VLADataLoader

from utils.similarity import VLASimilarity
from utils.metrics import VLAMetrics
from utils.interpolator import Interpolator
from utils.loss_logger import EvalLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_paths(path, size=16):
    shape = path.shape
    out = torch.zeros(tuple([size])+shape)
    for i in range(shape[1]):
        for j in range(shape[0]):
            p = path[j,i]
            if p > 0:
                u = (-p - p) * torch.rand(size) + p
            else:
                u = (p + p) * torch.rand(size) - p
            out[:,j,i] = u

    return out

def eval(clip, dataset, model_path):

    similarity = VLASimilarity()
    metrics = VLAMetrics()
    interpolator = Interpolator(96)

    mse_log = EvalLogger(model_path+"eval/", "mse-tuned")
    suc_log = EvalLogger(model_path+"eval/", "success-tuned")
    cls_log = EvalLogger(model_path+"eval/", "classification-tuned")

    count = 0
    avg_mse = 0
    suc_count = 0
    cls_count = 0

    for data in dataset:
        if data.text == None:
            continue

        path = data.path

        paths = generate_paths(data.path)
        
        img = data.image[0]
        txt_encoded = clip.encode_text(data.text[0]).squeeze()
        img_encoded = clip.encode_image(img[:,:,:-1]).squeeze()
        
        rand_ind = torch.randint(0,16,(1,))[0].item()
        paths[rand_ind] = torch.Tensor(data.path)

        pth_encoded = clip.encode_path(torch.Tensor(interpolator.interpolate_batch(paths))).squeeze()
        
        sim_it = similarity.get_1d_logits(img_encoded, txt_encoded)
        sim_tp = similarity.get_logits(txt_encoded, pth_encoded)
        sim_ip = similarity.get_logits(img_encoded, pth_encoded)

        sim =  (sim_tp + sim_ip) / 2
        maxs = torch.argmax(sim)
    
        path_t = torch.Tensor(path)
        pred = paths[maxs]
        mse = metrics.mse(pred, path_t).item()
        suc = metrics.success(pred, path_t)
        cls = metrics.classification(maxs, rand_ind)
        
        count += 1

        avg_mse = (avg_mse + mse) / count
        suc_count += suc
        cls_count += cls
        
        print("[VLA-EVAL] Avg. MSE: %s | success: %s/%s | classification: %s/%s" %(avg_mse,suc_count,count,cls_count,count))
        
    mse_log.log(avg_mse)
    suc_log.log(suc_count)
    suc_log.log(count)
    cls_log.log(cls_count)
    cls_log.log(count)


if __name__ == "__main__":

    data_path = "/home/jasonah/data/rxr-data/"
    model_path = "/home/jasonah/models/saved/" 

    dataset = VLADataset(data_path, mode='eval')

    txt_path = model_path + "clip_models/bert-tuned-7"
    img_path = model_path + "clip_models/vit-tuned-7"
    pth_path = "clip_models/path-tuned-7.pth"

    clip = CLIP3D('eval',
                  model_path,
                  img_model_path = img_path,
                  txt_model_path = txt_path,
                  pth_model_path = pth_path)

    eval(clip, dataset, model_path)
