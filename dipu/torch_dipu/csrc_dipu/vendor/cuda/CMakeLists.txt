# if torch compiled with cuda, find torch do find cuda. compile only cpu not.
find_package(CUDA "11.0" REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(NCCL REQUIRED)

if (CUDA_FOUND)
    message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
    message(STATUS "CUDA include: " ${CUDA_INCLUDE_DIRS})
    message(STATUS "NVCC Path: ${CUDA_TOOLKIT_ROOT_DIR}/bin")
    message(STATUS "NCCL include Path:  ${NCCL_INCLUDE_DIR}")
    message(STATUS "NCCL lib Path:  ${NCCL_LIBRARIES}")
endif()

set(VENDOR_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${NCCL_INCLUDE_DIR} PARENT_SCOPE)

# cuda is full lib path? libcudart_static.aThreads::Threadsdl/usr/lib64/librt.so
set(VENDOR_LIB_DIRS ${NCCL_LIBRARIES} PARENT_SCOPE)


if (CUDA_FOUND)
    file(GLOB PATCH_SRC_FILES  patch/*.cpp)
endif()

file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} ${PATCH_SRC_FILES} PARENT_SCOPE)

# now dipu not contains device code, but may be in future.
# include(cmake/CUDAComputeArch.cmake)
set(ARCH_HINTS)
set(CUDA_ARCH_AUTO ON)
if (CUDA_ARCH_AUTO)
    LIST(APPEND ARCH_HINTS Auto)
else()
    LIST(APPEND ARCH_HINTS Common)
endif()
cuda_select_nvcc_arch_flags(ARCH_FLAGS ${ARCH_HINTS})

message(STATUS "Selected CUDA archs : ${ARCH_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} --expt-extended-lambda)
# discuss: we try to build device code separately?
# cuda_add_library
