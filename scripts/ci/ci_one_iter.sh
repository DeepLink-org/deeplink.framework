#!/bin/bash

function clone_needed_repo() {
    # clone some repositories

    #define some version
    MMCV_VERSION=gqw/change_name
    MMENGINE_VERSION=v0.7.4
    MMPRETRAIN_VERSION=dipu_v1.0.0rc7_one_iter_tool
    MMDETECTION_VERSION=dipu_v3.0.0_one_iter_tool
    MMSEGMENTATION_VERSION=dipu_v1.0.0_one_iter_tool
    MMPOSE_VERSION=dipu_v1.0.0_one_iter_tool
    MMDETECTION3D_VERSION=dipu_v1.1.0_one_iter_tool
    MMACTION2_VERSION=dipu_v1.0.0_one_iter_tool
    MMOCR_VERSION=dipu_v1.0.0_one_iter_tool
    MMAGIC=dipu_v1.0.0_one_iter_tool
    SMART_VERSION=dev_for_mmcv2.0

    rm -rf SMART && git clone -b ${SMART_VERSION} https://github.com/ParrotsDL/SMART.git
    rm -rf mmpretrain && git clone -b ${MMPRETRAIN_VERSION} https://github.com/DeepLink-org/mmpretrain.git
    rm -rf mmdetection && git clone -b ${MMDETECTION_VERSION} https://github.com/DeepLink-org/mmdetection.git
    rm -rf mmsegmentation && git clone -b ${MMSEGMENTATION_VERSION} https://github.com/DeepLink-org/mmsegmentation.git
    rm -rf mmpose && git clone -b ${MMPOSE_VERSION} https://github.com/DeepLink-org/mmpose.git
    rm -rf mmdetection3d && git clone -b ${MMDETECTION3D_VERSION} https://github.com/DeepLink-org/mmdetection3d.git
    rm -rf mmaction2 && git clone -b ${MMACTION2_VERSION} https://github.com/DeepLink-org/mmaction2.git
    rm -rf mmocr && git clone -b ${MMOCR_VERSION} https://github.com/DeepLink-org/mmocr.git
    rm -rf mmagic && git clone -b ${MMAGIC} https://github.com/DeepLink-org/mmagic.git
    rm -rf mmengine && git clone -b ${MMENGINE_VERSION} https://github.com/open-mmlab/mmengine.git
    rm -rf mmcv && git clone -b ${MMCV_VERSION} https://github.com/Gong-air/mmcv.git
}

function build_needed_repo_cuda() {
    cd mmcv
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    cd ..
    cd mmagic
    pip install -e . -v --no-deps
    cd ..
}

function build_needed_repo_camb() {
    cd mmcv
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    cd ..
}


function export_repo_pythonpath(){
    basic_path="$2"
    if [ "$1" = "cuda" ]; then
        echo "Executing CUDA operation in pythonpath..."
        export PYTHONPATH=/nvme/share/share/platform/env/miniconda3.8/envs/pt2.0_diopi/mmcvs/9b1209f:$PYTHONPATH
        export PYTHONPATH=${basic_path}/mmagic:$PYTHONPATH
        export PYTHONPATH=${basic_path}/data/stable-diffusion-v1-5:$PYTHONPATH
        export PYTHONPATH=${basic_path}/mmagic/mmagic/models/editors/stable_diffusion:$PYTHONPATH
    elif [ "$1" = "camb" ]; then
        echo "Executing CAMB operation in pythonpath..."
        export PYTHONPATH=/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0_diopi/mmcvs/9b1209f:$PYTHONPATH
    else
        echo "Invalid parameter. Please specify 'cuda' or 'camb'."
        exit 1
    fi
    export PYTHONPATH=${basic_path}/mmpose:$PYTHONPATH
    export PYTHONPATH=${basic_path}/mmaction2:$PYTHONPATH
    export PYTHONPATH=${basic_path}/mmpretrain:$PYTHONPATH
    export PYTHONPATH=${basic_path}/mmocr:$PYTHONPATH
    export PYTHONPATH=${basic_path}/mmsegmentation:$PYTHONPATH
    export PYTHONPATH=${basic_path}/mmdetection3d:$PYTHONPATH
    export PYTHONPATH=${basic_path}/mmdetection:$PYTHONPATH
    export PYTHONPATH=${basic_path}/mmengine:$PYTHONPATH
    # export PYTHONPATH=${basic_path}/mmcv:$PYTHONPATH
    export PYTHONPATH=${basic_path}/SMART/tools/one_iter_tool/one_iter:$PYTHONPATH
    echo "python path: $PYTHONPATH"
}


function build_dataset(){
    # link dataset
    if [ "$1" = "cuda" ]; then
        echo "Executing CUDA operation in build dataset..."
        rm -rf data
        mkdir data
        ln -s /nvme/share/share_data/datasets/classification/imagenet data/imagenet
        ln -s /nvme/share/share_data/datasets/detection/coco  data/coco
        ln -s /nvme/share/share_data/datasets/segmentation/cityscapes data/cityscapes
        ln -s /nvme/share/share_data/datasets/detection3d/kitti data/kitti
        ln -s /nvme/share/share_data/chenwen/Kinetics400 data/kinetics400
        ln -s /nvme/share/share_data/chenwen/ocr/det/icdar2015/imgs data/icdar2015
        ln -s /nvme/share/share_data/datasets/ocr/recog/Syn90k/mnt/ramdisk/max/90kDICT32px data/mjsynth
        ln -s /nvme/share/share_data/slc/stable-diffusion-v1-5 data/stable-diffusion-v1-5
        ln -s /nvme/share/share_data/slc/swin_large_patch4_window12_384_22k.pth data/swin_large_patch4_window12_384_22k.pth
    elif [ "$1" = "camb" ]; then
        echo "Executing CAMB operation in build dataset..."
        rm -rf data
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
}


case $1 in
    clone)
        clone_needed_repo;;
    build_cuda)
        build_needed_repo_cuda
        build_dataset cuda;;
    build_camb)
        build_needed_repo_camb
        build_dataset camb;;
    export_pythonpath_camb)
        export_repo_pythonpath camb $2;;
    export_pythonpath_cuda)
        export_repo_pythonpath cuda $2;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac



