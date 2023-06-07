#!/bin/bash
#因为涉及到pythonpath的设置，因此请使用source来执行此脚本

function clone_needed_repo() {
    # clone some repositories

    #define some version
    MMCV_VERSION=v2.0.0
    MMENGINE_VERSION=v0.7.3
    MMPRETRAIN_VERSION=dipu_test_model_one_iter
    MMDETECTION_VERSION=one_iter_for_mmcv_2.0
    MMSEGMENTATION_VERSION=one_iter_for_mmcv_2.0
    MMPOSE_VERSION=one_iter_for_mmcv_2.0
    MMDETECTION3D_VERSION=one_iter_for_mmcv_2.0
    MMACTION2_VERSION=one_iter_for_mmcv_2.0
    MMOCR_VERSION=one_iter_for_mmcv_2.0
    SMART_VERSION=slc/test_one_iter

    rm -rf SMART && git clone -b ${SMART_VERSION} https://github.com/ParrotsDL/SMART.git
    rm -rf mmpretrain && git clone -b ${MMPRETRAIN_VERSION} https://github.com/DeepLink-org/mmpretrain.git
    rm -rf mmdetection && git clone -b ${MMDETECTION_VERSION} https://github.com/DeepLink-org/mmdetection.git
    rm -rf mmsegmentation && git clone -b ${MMSEGMENTATION_VERSION} https://github.com/DeepLink-org/mmsegmentation.git
    rm -rf mmpose && git clone -b ${MMPOSE_VERSION} https://github.com/DeepLink-org/mmpose.git
    rm -rf mmdetection3d && git clone -b ${MMDETECTION3D_VERSION} https://github.com/DeepLink-org/mmdetection3d.git
    rm -rf mmaction2 && git clone -b ${MMACTION2_VERSION} https://github.com/DeepLink-org/mmaction2.git
    rm -rf mmocr && git clone -b ${MMOCR_VERSION} https://github.com/DeepLink-org/mmocr.git
    rm -rf mmcv && git clone -b ${MMCV_VERSION} https://github.com/open-mmlab/mmcv.git
    rm -rf mmengine && git clone -b ${MMENGINE_VERSION} https://github.com/open-mmlab/mmengine.git
}

function build_needed_repo() {
    cd mmcv
    MMCV_WITH_OPS=1 python setup.py build_ext --inplace
    cd ..
}

function add_repo_pythonpath(){
    basic_path=$1
    # export PYTHONPATH=$basic_path/mmaction2:$PYTHONPATH
    # export PYTHONPATH=$basic_path/mmpretrain:$PYTHONPATH
    # export PYTHONPATH=$basic_path/mmocr:$PYTHONPATH
    # export PYTHONPATH=$basic_path/mmdetection3d:$PYTHONPATH
    # export PYTHONPATH=$basic_path/mmdetection:$PYTHONPATH
    export PYTHONPATH=$basic_path/mmengine:$PYTHONPATH
    export PYTHONPATH=$basic_path/mmcv:$PYTHONPATH
    export PYTHONPATH=$basic_path/SMART/tools/one_iter_tool/one_iter:$PYTHONPATH
}

function build_dataset(){
    # link dataset
    if [ "$1" = "cuda" ]; then
        echo "Executing CUDA operation..."
        mkdir data
        ln -s /nvme/share/share_data/datasets/classification/imagenet data/imagenet
        ln -s /nvme/share/share_data/datasets/detection/coco  data/coco
        ln -s /nvme/share/share_data/datasets/segmentation/cityscapes data/cityscapes
        ln -s /nvme/share/share_data/datasets/detection3d/kitti data/kitti
        ln -s /nvme/share/share_data/chenwen/Kinetics400 data/kinetics400
        ln -s /nvme/share/share_data/chenwen/ocr/det/icdar2015/imgs data/icdar2015     
        ln -s /nvme/share/share_data/datasets/ocr/recog/Syn90k/mnt/ramdisk/max/90kDICT32px data/mjsynth
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
}


case $1 in
    clone)
        (
            clone_needed_repo
        ) \
        || exit -1;;
    build_cuda)
        (
            build_needed_repo
            build_dataset cuda
        ) \
        || exit -1;;
    build_camb)
        (
            build_needed_repo
            build_dataset camb
        ) \
        || exit -1;;
    source_pythonpath)
        (
            add_repo_pythonpath $2
        ) \
        || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0

