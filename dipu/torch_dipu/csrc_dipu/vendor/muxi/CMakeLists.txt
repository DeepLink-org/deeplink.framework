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
if (CUDA_FOUND AND ${DIPU_TORCH_HAS_CUDA})
    set(SRC_FILES
        ../cudapatch/DIPUPatchCudaAllocator.cpp
        ../cudapatch/wrapperRegister.cpp)
endif()

set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/communiatorimpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/CUDACopyInplace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/CudaGeneratorImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/deviceimpl.cpp)

set(VENDOR_FILES  ${SRC_FILES} ${PATCH_SRC_FILES} PARENT_SCOPE)
