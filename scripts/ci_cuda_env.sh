export PYTORCH_DIR=${HOME}/pytorch/pytorch
export DIOPI_ROOT=$(pwd)/third_party/DIOPI/DIOPI-IMPL/lib/
export DIPU_ROOT=$(pwd)/torch_dipu
export LIBRARY_PATH=$DIPU_ROOT:${DIOPI_ROOT}:${LIBRARY_PATH}; LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH
export PATH=${CONDA_ROOT}/envs/dipu_poc/bin:${PATH}
export PYTHON_INCLUDE_DIR="/opt/conda/envs/py_3.10/include/python3.10/"