#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/abc_5K_rem2K \
--name abc_5K_rem2K \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 512 \
--ninput_edges 2000 \
--pool_res 1600 1280 1024 850 \
--resblocks 3 \
--lr 0.001 \
--batch_size 10 \
--num_aug 1 \
--continue_train
