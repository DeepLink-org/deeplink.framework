export GCC_ROOT=/mnt/lustre/share/platform/dep/gcc-7.5
export DIOPI_ROOT=$(pwd)/../DIOPI-TEST/lib/no_runtime
export DIPU_ROOT=$(pwd)/torch_dipu
export PYTORCH_DIR=/mnt/cache/share/parrotsci/github/cibuild/pytorch
export LIBRARY_PATH=$DIPU_ROOT:${DIOPI_ROOT}:${LIBRARY_PATH}; LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH
export PYTHONPATH=${PYTORCH_DIR}/install_path/lib/python3.8/site-packages:${PYTHONPATH}
export PATH=${PYTORCH_DIR}/install_path/bin:${PATH}
export LD_PRELOAD=${MPI_ROOT}/lib/libmpi.so:${GCC_ROOT}/lib64/libstdc++.so.6
export PYTHON_INCLUDE_DIR="/mnt/lustre/share/platform/env/miniconda3.8/envs/pt2.0v1_cpu/include/python3.8"
