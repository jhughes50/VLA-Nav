"""
    CIS 6200 -- Deep Learning Final Project
    Unit test for the habitat frame extractor
    April 2024
"""

import os 
import sys
import gzip
import json
import PIL 
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from preprocess.frame_extractor import ImageExtractor


def print_images(images):
    
    for i, arr in enumerate(images):
        img = Image.fromarray(arr)
        #img = img.rotate(270)
        img.save("test_%s.png"%i)


def main(path):
    im = ImageExtractor(path, 'train')

    counter = 0
    guide = list()
    with gzip.open(path+"rxr_train_guide.jsonl.gz") as f:
        print("found file %s, loading..." %path)
        for line in f:
            guide.append(json.loads(line))
            #if len(guide) == 2:
            #    break

    for subguide in guide:
        print(subguide)
        ##if subguide["instruction_id"] == 14:
        rgb = im.get_images( subguide )
        print("rgb: ", len(rgb))
        print_images(rgb)
        

if __name__ == "__main__":

    path = "/home/jasonah/data/rxr-data/" 
    main( path )

