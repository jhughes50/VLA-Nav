"""
    CIS 6200 -- Deep Learning Final Project
    Wrapper for our pretrained model encoders
    May 2024
"""

import torch
import yaml
from models.pose_encoder_small import PoseEncoder
from models.path_encoder_medium import PathEncoderMedium
from models.path_encoder_large import PathEncoderTransformer

class PathModelWrapper:

    def __init__(self, mode, input_path, size='large'):

        self.model_ = self.init_model(input_path, size)
        self.set_mode(mode)

    def model(self, path):
        return self.model_(path)

    def init_model(self, input_path, size):
        with open(input_path+"pretraining_config.yaml") as y:
            params = yaml.safe_load(input_path+"pretraining_config.yaml")
        
        params = params[size]

        if size == 'small':
            model = PoseEncoder(params["input_dim"],
                                params["hidden_dim"],
                                params["output_dim"])
            model.load_state_dict(torch.load(input_path+"%s_models/"%size))
        elif size == 'medium': 
            model = PathEncoderMedium(params["input_dim"],
                                      params["hidden_dim"],
                                      params["output_dim"])
            model.load_state_dict(torch.load(input_path+"%s_models/encoder_medium_epoch9.pth"%size))
        elif size == 'large':
            model = PathEncoderTransformer(params["input_dim"],
                                           params["output_dim"],
                                           params["model_dim"],
                                           params["num_heads"],
                                           params["hidden_dim"],
                                           params["num_layers"],
                                           params["dropout"])
            model.load_state_dict(torch.load(input_path+"%s_models/encoder_large_epoch2.pth"%size))
        else:
            print("[PATH-WRAPPER] Size %s is not defined, use small, medium or large" %size)
            exit()

        return model

    def set_mode(self, mode):
        if mode == 'train':
            self.model_.train()
        elif mode == 'eval':
            self.model_.eval()
        else:
            print("[PATH-WRAPPER] Mode %s is undefined, use train of eval" %mode)
