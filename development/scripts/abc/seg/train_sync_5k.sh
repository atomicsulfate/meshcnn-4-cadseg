#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/synth_dataset_5K_full \
--name synth_dataset_5K \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 5000 \
--pool_res 2500 1250 800 \
--resblocks 3 \
--lr 0.001 \
--batch_size 24 \
--num_aug 1

