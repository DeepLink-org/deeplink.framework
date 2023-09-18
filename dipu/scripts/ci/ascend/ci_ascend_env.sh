export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib
export DIPU_ROOT=$(pwd)/torch_dipu
export LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LIBRARY_PATH 
export LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH
export PYTORCH_DIR=${CONDA_PREFIX}/lib/python3.8/site-packages
export PYTHONPATH=$(pwd)/../pytorch:$(pwd):$PYTHONPATH
