#!/usr/bin/env bash

# LLAMA_MODEL_DIR=$1
# LLAMA_FINETUNE_DIR=$2
# STABLE_DIFFUSION_MODEL_DIR=$3

export DIPU_MOCK_CUDA=False
# export LLAMA_MODEL_DIR=$1
# export LLAMA_FINETUNE_DIR=$2
# export STABLE_DIFFUSION_MODEL_DIR=$3
export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,_softmax.out"
