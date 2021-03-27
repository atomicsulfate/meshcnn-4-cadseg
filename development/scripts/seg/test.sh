#!/usr/bin/env bash

## run the training
python test.py \
--dataroot datasets/abc_10K_dataset \
--name abc_10K_dataset \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 512 \
--ninput_edges 2000 \
--pool_res 1600 1280 1024 850 \
--resblocks 3 \
--lr 0.001 \
--batch_size 10 \
--num_aug 1 \
--num_threads 0 \
--export_folder meshes

