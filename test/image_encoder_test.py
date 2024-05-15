import torch
from transformers import AutoImageProcessor, ViTModel
from PIL import Image
import requests

test_image_path = ["/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000000.png",
"/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000001.png",
"/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000002.png"]

def image_encode(image_paths):

    image = [Image.open(image_path) for image_path in image_paths]

    # Initialize the feature extractor
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Preprocess the image and prepare tensor
    inputs = image_processor(images=image, return_tensors="pt")

    # Load the pre-trained ViT model
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Forward pass through the model to get encoded features
    with torch.no_grad():
        outputs = model(**inputs)

    #last hidden state
    last_hidden_states = outputs.last_hidden_state #(1, 197, 768)

    # Extract the features from the CLS token (assumed to be the first token in the sequence)
    cls_features = last_hidden_states[:, 0, :]
    print(cls_features.shape)

    return cls_features

features = image_encode(test_image_path)

# Define a linear layer to reduce dimensionality to 256
linear = torch.nn.Linear(features.size(1), 
512)

# Apply the linear layer to get reduced feat
embedding = linear(features)


