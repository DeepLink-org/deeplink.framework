# for conda
export PATH=${CONDA_ROOT}:$PATH
export LD_LIBRARY_PATH=${ENV_PATH}/lib:$LD_LIBRARY_PATH
export PATH=${ENV_PATH}/bin:$PATH

# gcc
export CC="${CC_PATH}/gcc"
export CXX="${CC_PATH}/g++"
export PATH=${CC_PATH}:$PATH

# for torch_dipu-stpu
export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib/
export DIPU_ROOT=$(pwd)/torch_dipu
export DIOPI_PATH=$(pwd)/third_party/DIOPI/proto
export DIPU_PATH=${DIPU_ROOT}
export LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LIBRARY_PATH
export LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH

echo $ENV_PATH
source activate $ENV_PATH
