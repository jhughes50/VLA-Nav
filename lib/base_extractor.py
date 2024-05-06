"""
    CIS 6200 -- Deep Learning Final Project
    Base Extractor class
    April 2024
"""

import json
import gzip

from lib.pose_extractor import PoseExtractor
from lib.frame_extractor import ImageExtractor
from lib.text_extractor import TextExtractor

from lib.input_data import InputData

class BaseExtractor(PoseExtractor, ImageExtractor, TextExtractor):

    def __init__(self, path, mode): 
        PoseExtractor.__init__(self, path, mode)
        ImageExtractor.__init__(self, path, mode)
    
        self.rxr_guide_ = None
        self.generic_path_ = path
        self.load(path, mode)

    def load(self, path, mode):
        if mode == 'train':
            path = path + "rxr_train_guide.jsonl.gz"
        else:
            path = path + "rxr_val_seen_guide.jsonl.gz"
        with gzip.open(path, 'r') as f:
            print("[BaseExtractor] found file %s, loading..." %path)
            self.rxr_guide_ = [json.loads(line) for line in f]
            print("[BaseExtractor] guide loaded")

    @property
    def guide(self):
        return self.rxr_guide_

    def extract(self, idx):
        subguide = self.rxr_guide_[idx]

        text = self.get_text(subguide)
        if text == None:
            print("[BASE-EXTRACTOR] text was not english, skipping...")
            return InputData(None, None, None)
        image = self.get_images(subguide)
        path = self.get_path(subguide)
        #path = self.interpolate(path)

        return InputData(text, image, path)



