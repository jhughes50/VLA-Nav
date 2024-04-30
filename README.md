## VLA-Nav
Embedding Vision-Language-Actions for learning robot navigation

## Running Docker
Build the image with `docker build --rm -t vla-docker .`. You'll have to change the username in the `Dockerfile`. Once it is built run with `./docker_run.sh`. Once again you'll need to change the file path after the second `-v` to where your data is. You can also configure what gpus you are using, use `--gpus all` to use all of them.

## Pretraining
We have two encoder-decoder pairs (small and large) that can be used to pretrain an path encoder to bring it to embedding space. These can be trained as an Autoencoder with the associated decoders. The small model is three layer feed-forward network for both the encoder and decoder. The larger model is a transformer based model with two encoder and decoder layers respectively each with two Multi-attention heads. To pretrain these models enter the docker image, and navigate to scripts and run `pretrain_<model_size>.py` where `model_size` is `large` or `small`. You may need to change the paths to files if your data is mounted differently. 

## Proposed Architecture
The pretraining architecture draws inspiration from CLIP, leveraging its principles to construct a robust framework. Initially, we execute a three-dimensional embedding process encompassing potential paths, images, and textual data. This approach adopts a 3D contrastive learning paradigm, facilitating the computation of cosine similarities across these three dimensions. By doing so, the model encapsulates intricate relationships between paths, images, and text, enhancing its ability to interpret and synthesize complex multimodal inputs.
![pretrain_architecture](https://github.com/jhughes50/VLA-Nav/assets/63807125/a3e73050-3046-43f6-ab23-3e070e256f2c)
_3D Contrastive Fine-tuning_


Inference involves two distinct stages. First, the model receives an image and accompanying text as input. Subsequently, it orchestrates a sophisticated analysis process, seeking to identify the most plausible path associated with the given input pair. This determination is achieved by strategically maximizing cosine similarity within the embedding space crafted by the pre-trained model. By referencing this embedding space, which encapsulates a wealth of semantic information learned during pretraining, the model effectively translates the multimodal input into a cohesive output, ultimately identifying the most probable path.
![inference_architecture](https://github.com/jhughes50/VLA-Nav/assets/63807125/18a68ab1-741a-4787-a82f-4b54b4202657)
_Inference Architecture for Best Path_
