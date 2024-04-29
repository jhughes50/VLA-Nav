FROM continuumio/miniconda3:latest AS miniconda
FROM nvidia/cudagl:11.4.2-base-ubuntu20.04
#FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# system depends
RUN apt-get install -y --no-install-recommends gcc cmake sudo
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip git python3-dev python3-opencv libglib2.0.0 wget python3-pybind11 vim
RUN apt-get install -y --no-install-recommends libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev

# Add user
ARG GID=1000
ARG UID=1000
ENV USER jasonah
RUN addgroup --gid $GID $USER 
RUN useradd --system --create-home --shell /bin/bash --groups sudo -p "$(openssl passwd -1 ${USER})" --uid $UID --gid $GID $USER
WORKDIR /home/$USER

# setup conda
COPY --from=miniconda /opt/conda /opt/conda
RUN chown -R $USER: /opt/conda 
#RUN chown -R $USER: /home/$USER
#USER $USER
# habitat conda install
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && conda init bash && conda create -n habitat -y python=3.9 cmake=3.14.0 && conda activate habitat && conda install habitat-sim headless -c conda-forge -c aihabitat -y && conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

#install CLIP-ViL dependencies
RUN pip3 install tqdm stanza tensorboardX openai-clip

#USER $USER
WORKDIR /home/$USER
CMD ["bash"]
