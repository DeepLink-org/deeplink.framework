ENV_PATH=/mnt/cache/share/platform/cienv
source ${ENV_PATH}/pt2.0_ci

export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib
export DIPU_ROOT=$(pwd)/torch_dipu
export LD_LIBRARY_PATH=${DIPU_ROOT}:${LD_LIBRARY_PATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}

# MMCV build requirements
export DIOPI_PATH=$(pwd)/third_party/DIOPI/proto
export DIPU_PATH=${DIPU_ROOT}

# PyTorch path related settings
export PYTORCH_DIR=${ASCEND_TORCH_DIR}
export PYTHONPATH=${PYTORCH_DIR}:${PYTHONPATH}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

ARCH=$(uname -m)
echo "Current Architecture: $ARCH"

# set CPLUS_INCLUDE_PATH
if [ "$ARCH" == "aarch64" ]; then
    export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include/:$CPLUS_INCLUDE_PATH
elif [ "$ARCH" == "x86_64" ]; then
    export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/include/:$CPLUS_INCLUDE_PATH
fi
