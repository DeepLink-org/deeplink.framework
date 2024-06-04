find_package(CUDA REQUIRED)

set(_XCCL_NAME MCCL)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/cmake)
find_package(XCCL REQUIRED)

set(VENDOR_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${XCCL_INCLUDE_DIR} PARENT_SCOPE)

set(VENDOR_LIB_DIRS ${XCCL_LIB_DIR} PARENT_SCOPE)

# make_maca add libs automatically, the find(cuda) here is only used to pass cmake config.
message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
set(DIPU_VENDOR_LIB mccl PARENT_SCOPE)

message(STATUS "DIPU_TORCH_HAS_CUDA = ${DIPU_TORCH_HAS_CUDA}")
set(BASE_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cuda")
if (CUDA_FOUND AND ${DIPU_TORCH_HAS_CUDA})
    set(PATCH_SRC_FILES
        ${BASE_CUDA_DIR}/patch/DIPUPatchCudaAllocator.cpp
        ${BASE_CUDA_DIR}/patch/wrapperRegister.cpp)
endif()

set(SRC_FILES
    ${BASE_CUDA_DIR}/communiatorimpl.cpp
    ${BASE_CUDA_DIR}/CUDACopyInplace.cpp
    ${BASE_CUDA_DIR}/CudaGeneratorImpl.cpp
    ${BASE_CUDA_DIR}/deviceimpl.cpp)

set(VENDOR_FILES  ${SRC_FILES} ${PATCH_SRC_FILES} PARENT_SCOPE)
