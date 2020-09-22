#!/bin/bash

## for interactive mode
##      ./run-docker.sh  x

if [ "$1" ] ; then
   interactive="-it"
fi

# sync up models & vocab
ln -f ../models/*  models/



docker run -u $(id -u):$(id -g)  $interactive --rm \
    -p 8000:8000 \
    -p 80:8000 \
    sujee/sotu-text-generator