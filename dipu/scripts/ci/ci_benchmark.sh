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
    SMART_VERSION=dev_for_mmcv2.0
    TRANSFORMERS=main
    LIGHTLLM=main
    DEEPLINKEXT=2a47138de420a0147e8de70685e628d3732135d7
    ALPACALORA=sco_benchmark_finetune

    check_and_clone_repository "SMART" ${SMART_VERSION}
    check_and_clone_repository "transformers" ${TRANSFORMERS}
    check_and_clone_repository "lightllm" ${LIGHTLLM}
    check_and_clone_repository "DeepLinkExt" ${DEEPLINKEXT}
    check_and_clone_repository "alpaca-lora" ${ALPACALORA}
    cd ..
}

function build_needed_repo_cuda() {
    cd mmcv
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    cd ..
    cd DeepLinkExt
    python setup.py build_ext -i
    cd ..
    cd alpaca-lora
    pip install -r requirements.txt
    cd ..
}

function build_needed_repo_camb() {
    cd mmcv
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    cd ..
}

function build_needed_repo_ascend() {
    cd mmcv
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i 
    cd ..
}

function build_needed_repo_kunlunxin() {
    echo "skip"
}


function export_repo_pythonpath(){
    basic_path="$2"
    if [ "$1" = "cuda" ]; then
        echo "Executing CUDA operation in pythonpath..."
        export PYTHONPATH=${basic_path}:$PYTHONPATH
        export PYTHONPATH=${basic_path}/transformers/src:$PYTHONPATH
        export PYTHONPATH=${basic_path}/lightllm:$PYTHONPATH

        # set the environment variable for the transformers repository
        export HF_HOME=${basic_path}/huggingface
        export HUGGINGFACE_HUB_CACHE=/mnt/lustre/share_data/PAT/datasets/hub

        export PYTHONPATH=${basic_path}/mmcv:$PYTHONPATH
        export PYTHONPATH=${basic_path}/SMART/tools/one_iter_tool/one_iter:$PYTHONPATH
        echo "python path: $PYTHONPATH"
    fi
}


function build_dataset(){
    # link dataset
    if [ "$1" = "cuda" ]; then
        echo "Executing CUDA operation in build dataset..."
        rm -rf data
        mkdir data
        ln -s /mnt/lustre/share_data/PAT/datasets/Imagenet data/imagenet
        ln -s /mnt/lustre/share_data/PAT/datasets/mscoco2017  data/coco
        ln -s /mnt/lustre/share_data/PAT/datasets/mmseg/cityscapes data/cityscapes
        ln -s /mnt/lustre/share_data/PAT/datasets/Kinetics400 data/kinetics400 
        ln -s /mnt/lustre/share_data/PAT/datasets/icdar2015 data/icdar2015
        ln -s /mnt/lustre/share_data/PAT/datasets/mjsynth data/mjsynth
        ln -s /mnt/lustre/share_data/PAT/datasets/kitti data/kitti
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet/checkpoint/swin_large_patch4_window12_384_22k.pth data/swin_large_patch4_window12_384_22k.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/stable-diffusion-v1-5 data/stable-diffusion-v1-5
        ln -s /mnt/lustre/share_data/PAT/datasets/llama_1B_oneiter  data/llama_1B_oneiter

    elif [ "$1" = "camb" ]; then
        echo "Executing CAMB operation in build dataset..."
        rm -rf data
        mkdir data
        ln -s /mnt/lustre/share_data/PAT/datasets/Imagenet data/imagenet
        ln -s /mnt/lustre/share_data/PAT/datasets/mscoco2017  data/coco
        ln -s /mnt/lustre/share_data/PAT/datasets/mmseg/cityscapes data/cityscapes
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet3d/mmdet3d_kitti data/kitti
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
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet/checkpoint/swin_large_patch4_window12_384_22k.pth data/swin_large_patch4_window12_384_22k.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/pretrain/torchvision/resnet50-0676ba61.pth data/resnet50-0676ba61.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet/pretrain/vgg16_caffe-292e1171.pth data/vgg16_caffe-292e1171.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/mmdet/pretrain/darknet53-a628ea1b.pth data/darknet53-a628ea1b.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/mmpose/pretrain/hrnet_w32-36af842e.pth data/hrnet_w32-36af842e.pth
        ln -s /mnt/lustre/share_data/PAT/datasets/pretrain/mmcv/resnet50_v1c-2cccc1ad.pth data/resnet50_v1c-2cccc1ad.pth
    elif [ "$1" = "ascend910b" ]; then
        echo "Executing ASCEND operation in build dataset..."
        rm -rf data
        mkdir data
        ln -s /mnt/cache/share/datasets/Imagenet data/imagenet
        ln -s /mnt/cache/share/datasets/mscoco2017  data/coco
        ln -s /mnt/cache/share/datasets/mmseg/cityscapes data/cityscapes
        ln -s /mnt/cache/share/datasets/kitti data/kitti
        ln -s /mnt/cache/share/datasets/mmaction/Kinetics400 data/kinetics400
        ln -s /mnt/cache/share/datasets/mmocr/icdar2015 data/icdar2015
        ln -s /mnt/cache/share/datasets/mmocr/mjsynth data/mjsynth
        ln -s /mnt/cache/share/datasets/mmdet/checkpoint/swin_large_patch4_window12_384_22k.pth data/swin_large_patch4_window12_384_22k.pth
        ln -s /mnt/cache/share/datasets/pretrain/torchvision/resnet50-0676ba61.pth data/resnet50-0676ba61.pth
        ln -s /mnt/cache/share/datasets/mmdet/pretrain/vgg16_caffe-292e1171.pth data/vgg16_caffe-292e1171.pth
        ln -s /mnt/cache/share/datasets/mmdet/pretrain/darknet53-a628ea1b.pth data/darknet53-a628ea1b.pth
        ln -s /mnt/cache/share/datasets/mmpose/pretrain/hrnet_w32-36af842e.pth data/hrnet_w32-36af842e.pth
        ln -s /mnt/cache/share/datasets/pretrain/mmcv/resnet50_v1c-2cccc1ad.pth data/resnet50_v1c-2cccc1ad.pth
    elif [ "$1" = "kunlunxin" ]; then
        echo "Executing KUNLUNXIN operation in build dataset..."
        rm -rf data
        mkdir data
        ln -s /mnt/cache/share/datasets/imagenet data/imagenet

    else
        echo "Invalid parameter. Please specify 'cuda' 'camb' 'ascend' or 'kunlunxin'."
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
    build_ascend910b)
        build_needed_repo_ascend
        build_dataset ascend910b;;
    build_kunlunxin)
        build_needed_repo_kunlunxin
        build_dataset kunlunxin;;
    export_pythonpath_camb)
        export_repo_pythonpath camb $2;;
    export_pythonpath_cuda)
        export_repo_pythonpath cuda $2;;
    export_pythonpath_ascend)
        export_repo_pythonpath ascend $2;;
    export_pythonpath_kunlunxin)
        export_repo_pythonpath kunlunxin $2;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
