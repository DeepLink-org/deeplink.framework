#!/bin/bash

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
MMPRETRAIN_VERSION=dipu_v1.0.0rc7_one_iter_tool
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

