# ENV_NAME=dipu_poc
# export PATH=`python ${PLATFORM}/env/clear_path.py PATH`
# export LD_LIBRARY_PATH=`python ${PLATFORM}/env/clear_path.py LD_LIBRARY_PATH`
# GCC_ROOT=/mnt/lustre/share/platform/dep/gcc-7.5
export CONDA_ROOT=/home/cse/miniconda3
export PYTORCH_DIR=/home/cse/src/pytorch
# export CC=${GCC_ROOT}/bin/gcc
# export CXX=${GCC_ROOT}/bin/g++

export DIOPI_ROOT=/home/cse/src/deeplink/dipu/third_party/DIOPI/DIOPI-IMPL/lib
export DIPU_ROOT=$(pwd)/torch_dipu
export LIBRARY_PATH=$DIPU_ROOT:${DIOPI_ROOT}:${LIBRARY_PATH}; LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH
export PYTHONPATH=${CONDA_ROOT}/envs/dipu/lib/python3.8:${PYTHONPATH}:
export PATH=${PYTORCH_DIR}/build/bin:${CONDA_ROOT}/envs/dipu/bin:${CONDA_ROOT}/bin:${PATH}
# export LD_PRELOAD=${GCC_ROOT}/lib64/libstdc++.so.6
export PYTHON_INCLUDE_DIR="${CONDA_ROOT}/envs/dipu/include/python3.8"

# source activate $ENV_NAME
