#!/usr/bin/env bash

LLAMA_MODEL_DIR=$1

export DIPU_MOCK_CUDA=false
export LLAMA_MODEL_DIR=$1
export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"
