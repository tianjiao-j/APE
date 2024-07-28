#!/bin/bash

export START_TIME="`date +%Y_%m_%d-%H_%M`"
echo $START_TIME

# dtd oxford_pets stanford_cars ucf101 food101 sun397 fgvc eurosat oxford_flowers caltech101 imagenet
for DATASET_NAME in dtd oxford_pets stanford_cars ucf101 food101 sun397 fgvc eurosat oxford_flowers caltech101 imagenet
do
  for N_SHOT in 1 2 4 8 16
  do
    python main.py --config configs/${DATASET_NAME}.yaml --shot ${N_SHOT}
  done
done