#!/bin/bash

# link dataset

if [ "$1" = "cuda" ]; then
    echo "Executing CUDA operation..."
    
elif [ "$1" = "camb" ]; then
    echo "Executing CAMB operation..."
    mkdir data
    ln -s /mnt/lustre/share_data/PAT/datasets/Imagenet data/imagenet
    ln -s /mnt/lustre/share_data/PAT/datasets/mscoco2017  data/coco
    ln -s /mnt/lustre/share_data/PAT/datasets/mmseg/cityscapes data/cityscapes
    ln -s /mnt/lustre/share_data/slc/mmdet3d/mmdet3d data/kitti
    ln -s /mnt/lustre/share_data/PAT/datasets/mmaction/Kinetics400 data/kinetics400
    ln -s /mnt/lustre/share_data/PAT/datasets/mmocr/icdar2015 data/icdar2015
else
    echo "Invalid parameter. Please specify 'cuda' or 'camb'."
    exit 1
fi