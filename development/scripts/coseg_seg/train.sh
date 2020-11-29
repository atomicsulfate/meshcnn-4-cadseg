#!/usr/bin/env bash

## run the training
python train.py \
--dataroot meshcnn/datasets/coseg_aliens \
--name coseg_aliens \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 2280 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--lr 0.001 \
--batch_size 12 \
--num_aug 20 \
--slide_verts 0.2 \

