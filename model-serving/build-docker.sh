#!/bin/bash

# sync up models & vocab
ln -f ../models/*  models/
ln -f ../tokenizer-vocabulary/*  models/

docker build . -t sujee/sotu-text-generator
