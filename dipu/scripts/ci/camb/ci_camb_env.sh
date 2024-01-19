PLATFORM=/mnt/lustre/share/platform
ENV_NAME=pt2.0_diopi
export PATH=`python ${PLATFORM}/env/clear_path.py PATH`
export LD_LIBRARY_PATH=`python ${PLATFORM}/env/clear_path.py LD_LIBRARY_PATH`
GCC_ROOT=/mnt/lustre/share/platform/dep/gcc-10.2
CONDA_ROOT=${PLATFORM}/env/miniconda3.10

export NEUWARE_HOME=/usr/local/neuware
export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

export DIOPI_ROOT=$(pwd)/third_party/DIOPI/impl/lib/
export DIPU_ROOT=$(pwd)/torch_dipu
export LD_LIBRARY_PATH=$DIPU_ROOT:$LD_LIBRARY_PATH
if [ -n "$1" ];then
  export PYTHONPATH=${PLATFORM}/dep/DIOPI_pytorch/pytorch$1:${PYTHONPATH}
else
  export PYTHONPATH=${PLATFORM}/dep/DIOPI_pytorch/pytorch2.0:${PYTHONPATH}
fi
export PATH=${GCC_ROOT}/bin:${CONDA_ROOT}/envs/dipu_poc/bin:${CONDA_ROOT}/bin:${PATH}
export LD_PRELOAD=${GCC_ROOT}/lib64/libstdc++.so.6

export NEUWARE_ROOT_DIR=${NEUWARE_HOME}
export VENDOR_INCLUDE_DIRS=${NEUWARE_HOME}/include
export DIOPI_PATH=$(pwd)/third_party/DIOPI/proto
export DIPU_PATH=${DIPU_ROOT}

#export MLU_INVOKE_BLOCKING=1
#export DIPU_DEBUG_ALLOCATOR=15
export DIPU_DEVICE_MEMCACHING_ALGORITHM=BS
export DIPU_HOST_MEMCACHING_ALGORITHM=BS
#export DIPU_BS_ALLOCATOR_MIN_ALLOCATE_SIZE=512
#export DIPU_RAW_ALLOCATOR_MIN_ALLOCATE_SIZE=512
export DIPU_CHECK_TENSOR_DEVICE=1

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

source activate $ENV_NAME

echo  "python path : ${PYTHONPATH}"
