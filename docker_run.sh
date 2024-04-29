#!/bin/sh
docker run --rm -it -e "TERM=xterm-256color" --gpus '"device=0"' -v ".:/home/jasonah" vla-docker bash 
