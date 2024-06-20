# waitting ci env ready & add new env var.......


export DEEPLINK_HOME=$(pwd)
export DIPU_HOME="${DEEPLINK_HOME}/dipu"
export LD_LIBRARY_PATH="${DIPU_HOME}/torch_dipu:${LD_LIBRARY_PATH}"
export PYTHONPATH=":${DIPU_HOME}:${PYTHONPATH}"
export DIOPI_ROOT=${DIPU_HOME}/third_party/DIOPI/impl/lib/


export DIPU_CUDA_EVENT_TIMING=1
export DIPU_DEVICE_MEMCACHING_ALGORITHM=TORCH
export DIPU_HOST_MEMCACHING_ALGORITHM=TORCH
export DIPU_PATCH_CUDA_CACHED_ALLOCATOR=0
export DIPU_CHECK_TENSOR_DEVICE=1

# Setting OMP_NUM_THREADS environment variable for each process in default,
# to avoid your system being overloaded, please further tune the variable
# for optimal performance in your application as needed.
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# TODO: add muxi ci specific variable.

# source activate $ENV_NAME