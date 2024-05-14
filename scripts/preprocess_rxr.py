"""
    CIS 6200 -- Deep Learning Final Project
    Extract and save images and paths for training
    May 2024
"""

import os
import sys
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.base_extractor import BaseExtractor

READ_PATH = '/home/jasonah/data/rxr-data/'
WRITE_PATH = '/home/jasonah/data/rxr-data/'

def genpath(path, idx):
    if idx == 0:
        return path
    else:
        return path[idx:]


def save(iid, vpids, images, path):
    assert len(vpids) >= path.shape[0],\
        "viewpoint should be >= waypoints, but got %s, %s, with %s images" %(len(vpids), path.shape[0], len(images))
   
    print("[PREPROCESS] Num images: %s, Num Viewpoints: %s, Num WPs: %s" %(len(images), len(vpids), path.shape[0]))
    for i, img in enumerate(images):
        img = img[:,:,:-1]
        pimg = Image.fromarray(img.astype('uint8'), 'RGB')
        pimg.save(WRITE_PATH+"/{:06}_{}.jpg".format(iid, vpids[i]))
        subpath = genpath(path, i)
        np.savetxt(WRITE_PATH+"/{:06}_{}.txt".format(iid, vpids[i]), subpath)
        

def extract(base):
    
    for idx in range(len(base.guide)):
        data = base.extract(idx)
        if data.text != None:
            instr_id = base.guide[idx]["instruction_id"]
            vpids = base.guide[idx]["path"]
            save(instr_id, vpids, data.image, data.path)        


if __name__ == "__main__":
    # select dataset to extract
    mode = 'train'
    
    # create directory for data dump
    WRITE_PATH = WRITE_PATH + mode
    if not os.path.exists(WRITE_PATH):
        os.makedirs(WRITE_PATH)

    # instantiate the extractor 
    base = BaseExtractor(READ_PATH, mode)

    extract(base)
