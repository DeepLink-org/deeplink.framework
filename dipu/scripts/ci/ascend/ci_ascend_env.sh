ENV_PATH=/mnt/cache/share/platform/cienv
source ${ENV_PATH}/pt2.0_ci

export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib
export DIPU_ROOT=$(pwd)/torch_dipu
export LIBRARY_PATH=${DIPU_ROOT}:${DIOPI_ROOT}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${DIPU_ROOT}:${DIOPI_ROOT}:${LD_LIBRARY_PATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}

# MMCV build requirements
export DIOPI_PATH=$(pwd)/third_party/DIOPI/proto
export DIPU_PATH=${DIPU_ROOT}

# PyTorch path related settings
export PYTORCH_DIR=${ASCEND_TORCH_DIR}
export PYTHONPATH=${PYTORCH_DIR}:${PYTHONPATH}
