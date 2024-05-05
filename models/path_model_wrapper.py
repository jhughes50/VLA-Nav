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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PathModelWrapper:

    def __init__(self, mode, input_path, eval_model, size='medium'):

        if eval_model == None:
            self.model_ = self.init_model(input_path, size)
        else:
            self.model_ = self.init_pretrained(input_path, model_path)

        self.set_mode(mode)
        self.size_ = size

        self.model_.to(DEVICE)
        print("[PATH-WRAPPER] model on cuda: ", next(self.model_.parameters()).is_cuda)

    def model(self, path):
        if self.size_ != 'large':
            path = path.reshape((16,96))
        return self.model_(path)

    def init_pretrained(self, input_path, model_path):
        with open(input_path+"pretraining_config.yaml") as y:
            params = yaml.safe_load(y)

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
            model.load_state_dict(torch.load(input_path+model_path))
        elif size == 'large':
            model = PathEncoderTransformer(params["input_dim"],
                                           params["output_dim"],
                                           params["model_dim"],
                                           params["num_heads"],
                                           params["hidden_dim"],
                                           params["num_layers"],
                                           params["dropout"])
            model.load_state_dict(torch.load(input_path+model_path))
        else:
            print("[PATH-WRAPPER] Size %s is not defined, use small, medium or large" %size)
            exit()

    def init_model(self, input_path, size):
        with open(input_path+"pretraining_config.yaml") as y:
            params = yaml.safe_load(y)

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
            model.load_state_dict(torch.load(input_path+"%s_models/encoder_large_epoch3.pth"%size))
        else:
            print("[PATH-WRAPPER] Size %s is not defined, use small, medium or large" %size)
            exit()

        return model

    def get_params(self):
        return list(self.model_.parameters())

    def set_mode(self, mode):
        if mode == 'train':
            self.model_.train()
        elif mode == 'eval':
            self.model_.eval()
        else:
            print("[PATH-WRAPPER] Mode %s is undefined, use train of eval" %mode)

    def save(self, output_dir, idx):
        print("[PATH-WRAPPER] Saving model to %s at index %s" %(output_dir, idx))
        torch.save(self.model_.state_dict(), output_dir+"path-tuned-%s.pth"%idx)
