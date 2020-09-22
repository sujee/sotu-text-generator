#!/bin/bash

# sync up models & vocab
rm -rf models/*
ln -f ../models/*model-1.h5  models/
ln -f ../tokenizer-vocabulary/*  models/

docker build . -t sujee/sotu-text-generator
