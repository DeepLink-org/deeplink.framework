ENV_PATH=/mnt/cache/share/platform/cienv
source ${ENV_PATH}/pt2.0_ci

export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib
export DIPU_ROOT=$(pwd)/torch_dipu
export LIBRARY_PATH=${DIPU_ROOT}:${DIOPI_ROOT}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${DIPU_ROOT}:${DIOPI_ROOT}:${LD_LIBRARY_PATH}
