#!/bin/bash

# sync up models & vocab
ln -f ../models/*  models/

docker build . -t sujee/sotu-text-generator
