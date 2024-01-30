#!/usr/bin/env bash

ENV_PATH=/mnt/cache/share/platform/cienv
source ${ENV_PATH}/pt2.0_ci

export DIPU_ROOT=$(pwd)/dipu/torch_dipu
export LD_LIBRARY_PATH=${DIPU_ROOT}:${LD_LIBRARY_PATH}
export PYTHONPATH=${DIPU_ROOT}:${PYTHONPATH}

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh

ARCH=$(uname -m)
echo "Current Architecture: $ARCH"

# set CPLUS_INCLUDE_PATH
if [ "$ARCH" == "aarch64" ]; then
    export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include/:$CPLUS_INCLUDE_PATH
elif [ "$ARCH" == "x86_64" ]; then
    export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/include/:$CPLUS_INCLUDE_PATH
fi
