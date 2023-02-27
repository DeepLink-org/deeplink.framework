find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
    message(STATUS "cudnn libraries: " ${CUDNN_LIBRARIES})
    message(STATUS "CUDA include: " ${CUDA_INCLUDE_DIRS})
    message(STATUS "NVCC Path: ${CUDA_TOOLKIT_ROOT_DIR}/bin")
endif()

# set(CUDA_SEPARABLE_COMPILATION ON)

# -std=c++14  -G; -g;
# v100 is arch 70
set(CUDA_NVCC_FLAGS
    -Xcompiler '-fPIC'
    -gencode arch=compute_70,code=sm_70
    -I ${ROOT}/cuda/include
    -L ${ROOT}/cuda/lib64)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g  -O0")

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64/)

FILE(GLOB VENDOR_FILES
    test_cuda1.cpp
)
 