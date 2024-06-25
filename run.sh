#!/bin/bash

for DATASET_NAME in dtd oxford_pets stanford_cars ucf101 food101 sun397 fgvc_aircraft eurosat oxford_flowers imagenet
do
  for N_SHOT in 1 2 4 8 16
  do
    python main.py --config configs/${DATASET_NAME}.yaml --shot ${N_SHOT}
  done
done