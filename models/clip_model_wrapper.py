"""
    CIS 6200 -- Deep Learning Final Project
    Wrapper for huggingface CLIP model from OpenAI
    May 2024
"""

from transformers import CLIPProcessor
from transformers import CLIPTokenizer
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPTextModelWithProjection

class CLIPWrapper:

    def __init__(self, mode, device, model_path = None):

        if model_path == None:
            self.img_model_ = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.txt_model_ = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

        self.img_model_.to(device)
        self.txt_model_.to(device)

        self.device_ = device

        self.processor_ = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer_ = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        print("[CLIP-WRAPPER] Image model on cuda: ", next(self.img_model_.parameters()).is_cuda)
        print("[CLIP-WRAPPER] Text model on cuda: ", next(self.txt_model_.parameters()).is_cuda)
    def set_mode(self, mode):
        if mode == 'train':
            self.model_.train()
        elif mode == 'eval' or mode == 'test':
            self.model_.eval()
        else:
            print("[CLIP-WRAPPER] Model mode must be train, eval or test not %s" %mode)
            exit()

    def model(self, text, image):
        txt_tokens = self.tokenizer_(text, truncation=True, return_tensors="pt")
        img_tokens = self.processor_(images=image, return_tensors="pt")

        txt_tokens = txt_tokens.to(self.device_)
        img_tokens = img_tokens.to(self.device_)

        txt_outputs = self.txt_model_(**txt_tokens)
        img_outputs = self.img_model_(**img_tokens)

        return txt_outputs.text_embeds, img_outputs.image_embeds

    def get_params(self):
        pass

