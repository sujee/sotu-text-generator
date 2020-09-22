#!/bin/bash

# sync up models & vocab
ln -f ../models/*  models/


docker run -u $(id -u):$(id -g)  -it --rm \
    -p 8000:8000 \
    sujee/sotu-text-generator