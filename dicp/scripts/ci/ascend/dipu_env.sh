#!/usr/bin/env bash

export DIPU_DEVICE=ascend
export DIPU_WITH_DIOPI_LIBRARY=DISABLE
export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"
