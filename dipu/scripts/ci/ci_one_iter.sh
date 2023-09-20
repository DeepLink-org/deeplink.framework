#!/bin/bash

function check_and_clone_repository() {
    repo_name=$1
    branch_name=$2
    current_path=$(pwd)
    repo_path="$current_path/$repo_name"
    if [ "$repo_name" == "mmcv" ] || [ "$repo_name" == "mmengine" ]; then
        clone_url="https://github.com/open-mmlab/$repo_name.git"
    else
        clone_url="https://github.com/DeepLink-org/$repo_name.git"
    fi
    if [ -d "$repo_path" ]; then
        cd $repo_name
        current_branch=$(git rev-parse --abbrev-ref HEAD)_$(git rev-parse HEAD)_$(git describe --tags 2>/dev/null || echo "none")
        if [[ "$current_branch" =~ "$branch_name" ]]; then
            echo "$repo_name $branch_name is right"
            cd ..
        else
            git checkout main && git pull && git checkout $branch_name 
            cd ..
        fi
    else
        cd $current_path && rm -rf  $repo_name
        git clone -b ${branch_name} ${clone_url} || (git clone ${clone_url} && cd $repo_name && git checkout ${branch_name} && cd ..)
    fi
}

function clone_needed_repo() {
    set -e
    # clone some repositories
    MMCV_VERSION=ee93530acc675231014b92a58fd6e4a59e27cc13
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
    MMYOLO=dipu_v0.5.0_one_iter_tool
    DIENGINE=dipu_v0.4.8_one_iter_tool
    TRANSFORMERS=dipu_v4.28.1_one_iter_tool

    check_and_clone_repository "DI-engine" ${DIENGINE}
    check_and_clone_repository "SMART" ${SMART_VERSION}
    check_and_clone_repository "mmpretrain" ${MMPRETRAIN_VERSION}
    check_and_clone_repository "mmdetection" ${MMDETECTION_VERSION}
    check_and_clone_repository "mmsegmentation" ${MMSEGMENTATION_VERSION}
    check_and_clone_repository "mmpose" ${MMPOSE_VERSION}
    check_and_clone_repository "mmdetection3d" ${MMDETECTION3D_VERSION}
    check_and_clone_repository "mmaction2" ${MMACTION2_VERSION}
    check_and_clone_repository "mmocr" ${MMOCR_VERSION}
    check_and_clone_repository "mmagic" ${MMAGIC}
    check_and_clone_repository "mmyolo" ${MMYOLO}
    check_and_clone_repository "mmengine" ${MMENGINE_VERSION}
    check_and_clone_repository "transformers" ${TRANSFORMERS}
    check_and_clone_repository "mmcv" ${MMCV_VERSION}
    cd ..
}

function build_needed_repo_cuda() {
    cd mmcv
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    cd ..
    # cd ../mmdet 
    # pip install -e . --no-deps
    # cd ../mmyolo
    # # Install albumentations
    # pip install -r requirements/albu.txt --no-deps
    # # Install MMYOLO
    # pip install -e . --no-deps
    # cd mmagic
    # pip install -e . -v 
    # cd ../mmpretrain
    # pip install -e .
    # cd ..
    # cd DI-engine
    # pip install -e .
    # cd ..
    # #安装强化学习需要用的包
    # pip install lz4
    # pip install readerwriterlock
    # pip install Flask==2.1.0
    # pip install transformers
    # pip install accelerate
}

function build_needed_repo_camb() {
    cd mmcv
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    cd ..
}

function build_needed_repo_ascend() {
    cd mmcv
    MMCV_VERSION_FOR_ONE_ITER=9b1209f
    if [ "$(git rev-parse HEAD)" != "${MMCV_VERSION_FOR_ONE_ITER}" ]; then
        git checkout ${MMCV_VERSION_FOR_ONE_ITER}
        rm -rf build
    fi
    if [ ! -d "build" ]; then
        MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i 
    fi
    cd ..
}


function export_repo_pythonpath(){
    basic_path="$2"
    if [ "$1" = "cuda" ]; then
        echo "Executing CUDA operation in pythonpath..."
        export PYTHONPATH=/mnt/cache/share/platform/env/miniconda3.8/envs/pt2.0_diopi/mmcvs/9b1209f:$PYTHONPATH
        export PYTHONPATH=${basic_path}/mmagic:$PYTHONPATH
        export PYTHONPATH=${basic_path}/data/stable-diffusion-v1-5:$PYTHONPATH
        export PYTHONPATH=${basic_path}/mmagic/mmagic/models/editors/stable_diffusion:$PYTHONPATH
    elif [ "$1" = "camb" ]; then
        echo "Executing CAMB operation in pythonpath..."
        export PYTHONPATH=/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0_diopi/mmcvs/9b1209f:$PYTHONPATH
        export PYTHONPATH=${basic_path}/mmagic:$PYTHONPATH
        export PYTHONPATH=${basic_path}/data/stable-diffusion-v1-5:$PYTHONPATH
        export PYTHONPATH=${basic_path}/mmagic/mmagic/models/editors/stable_diffusion:$PYTHONPATH
    elif [ "$1" = "ascend" ]; then
        echo "Executing ASCEND operation in pythonpath..."
        export PYTHONPATH=${basic_path}/mmcv:$PYTHONPATH
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
    export PYTHONPATH=${basic_path}/mmyolo:$PYTHONPATH
    export PYTHONPATH=${basic_path}/DI-engine:$PYTHONPATH
    export PYTHONPATH=${basic_path}/transformers:$PYTHONPATH

    # set the environment variable for the transformers repository
    export HF_HOME=${basic_path}/huggingface
    export HUGGINGFACE_HUB_CACHE=/mnt/lustre/share_data/PAT/datasets/hub

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
        ln -s /mnt/lustre/share_data/parrots.tester.s.03/dataset/data_for_ln/imagenet data/imagenet
        ln -s /mnt/lustre/share_data/parrots.tester.s.03/dataset/data_for_ln/coco  data/coco
        ln -s /mnt/lustre/share_data/parrots.tester.s.03/dataset/data_for_ln/cityscapes data/cityscapes
        ln -s /mnt/lustre/share_data/openmmlab/datasets/action/Kinetics400 data/kinetics400 
        ln -s /mnt/lustre/share_data/parrots.tester.s.03/dataset/data_for_ln/icdar2015 data/icdar2015
        ln -s /mnt/lustre/share_data/parrots.tester.s.03/dataset/data_for_ln/mjsynth data/mjsynth
        ln -s /mnt/lustre/share_data/parrots.tester.s.03/dataset/data_for_ln/kitti data/kitti
        ln -s /mnt/lustre/share_data/shenliancheng/swin_large_patch4_window12_384_22k.pth data/swin_large_patch4_window12_384_22k.pth
        ln -s /mnt/lustre/share_data/parrots.tester.s.03/models_code/mmagic/stable-diffusion-v1-5 data/stable-diffusion-v1-5

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
        ln -s /mnt/lustre/share_data/PAT/datasets/mmocr/mjsynth data/mjsynth
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet/checkpoint/swin_large_patch4_window12_384_22k.pth data/swin_large_patch4_window12_384_22k.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/pretrain/torchvision/resnet50-0676ba61.pth data/resnet50-0676ba61.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet/pretrain/vgg16_caffe-292e1171.pth data/vgg16_caffe-292e1171.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet/pretrain/darknet53-a628ea1b.pth data/darknet53-a628ea1b.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/mmpose/pretrain/hrnet_w32-36af842e.pth data/hrnet_w32-36af842e.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/pretrain/mmcv/resnet50_v1c-2cccc1ad.pth data/resnet50_v1c-2cccc1ad.pth
    
    elif [ "$1" = "ascend" ]; then
        echo "Executing ASCEND operation in build dataset..."
        rm -rf data
        mkdir data
        ln -s /mnt/lustre/share_data/PAT/datasets/Imagenet data/imagenet
        ln -s /mnt/lustre/share_data/PAT/datasets/mscoco2017  data/coco
        ln -s /mnt/lustre/share_data/PAT/datasets/mmseg/cityscapes data/cityscapes
        ln -s /mnt/lustre/share_data/PAT/datasets/kitti data/kitti
        ln -s /mnt/lustre/share_data/PAT/datasets/mmaction/Kinetics400 data/kinetics400
        ln -s /mnt/lustre/share_data/PAT/datasets/mmocr/icdar2015 data/icdar2015
        ln -s /mnt/lustre/share_data/PAT/datasets/mmocr/mjsynth data/mjsynth

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
    build_ascend)
        build_needed_repo_ascend
        build_dataset ascend;;
    export_pythonpath_camb)
        export_repo_pythonpath camb $2;;
    export_pythonpath_cuda)
        export_repo_pythonpath cuda $2;;
    export_pythonpath_ascend)
        export_repo_pythonpath ascend $2;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
