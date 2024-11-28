#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export START_TIME="`date +%Y_%m_%d-%H_%M`"
echo $START_TIME

# dtd oxford_pets stanford_cars ucf101 food101 sun397 fgvc eurosat oxford_flowers caltech101 imagenet
for DATASET_NAME in imagenet sun397 food101 stanford_cars caltech101 fgvc ucf101 oxford_pets oxford_flowers dtd eurosat
do
  for N_SHOT in 16 8 4 2 1
  do
    python main.py --config configs/${DATASET_NAME}.yaml --shot ${N_SHOT}
  done
done