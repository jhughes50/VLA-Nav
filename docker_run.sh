#!/bin/sh
docker run --rm -it -e "TERM=xterm-256color" --gpus '"device=0"' -v "/home/ec2-user/VLA-Nav:/home/jasonah" -v "/home/ec2-user/VLA-Nav-Data:/home/jasonah/data" vla-docker bash 
