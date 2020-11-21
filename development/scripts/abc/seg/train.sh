#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/abc_dummy \
--name coseg_aliens \
--arch meshunet \
--dataset_mode segmentation \
--fake_segdata \
--ncf 32 64 128 256 \
--ninput_edges 65000 \
--pool_res 32000 8000 1000 \
--resblocks 3 \
--lr 0.001 \
--batch_size 12 \
--num_aug 20 \
--slide_verts 0.2 \


