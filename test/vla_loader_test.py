"""
    CIS 6200 Final Project
    Unit test to test path extraction from the dataset. 
    April 2024
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(PROJECT_ROOT)

from lib.vla_dataloader import VLADataLoader

def main(path):

    dataloader = VLADataLoader(path)

    for x in dataloader:
        print(len(x.image))
        print(x.text)
        print(x.poses)
    

if __name__ == "__main__":

    path = "/home/jasonah/data/rxr-data/"
    main( path )

