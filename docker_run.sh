#!/bin/sh
docker run --rm -it -e "TERM=xterm-256color" --gpus '"device=0"' -v "/raid0/docker-raid/jasonah:/home/jasonah/data" -v ".:/home/jasonah" vla-docker bash
