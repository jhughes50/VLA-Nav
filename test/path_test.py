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

from preprocess.pose_extractor import PoseExtractor

def main(path):
    pe = PoseExtractor(path, 'train')
    pe.load()
    print(len(pe.train_guide_))
    #path = pe.path_from_guide(23)
    path = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]])
    path = pe.interpolate(path)
    print(path)
    print(path.shape)

if __name__ == "__main__":

    path = "/home/jasonah/data/"
    main( path )

