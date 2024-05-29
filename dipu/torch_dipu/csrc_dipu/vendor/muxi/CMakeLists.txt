find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
    set(SRC_FILES
        ../cudapatch/DIPUPatchCudaAllocator.cpp
        ../cudapatch/wrapperRegister.cpp)
endif()

set(_XCCL_NAME MCCL)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/cmake)
find_package(XCCL REQUIRED)

set(VENDOR_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${XCCL_INCLUDE_DIR} PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${XCCL_LIB_DIR} PARENT_SCOPE)
set(DIPU_VENDOR_LIB mccl PARENT_SCOPE)


set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/communiatorimpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/CUDACopyInplace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/CudaGeneratorImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/deviceimpl.cpp)

set(VENDOR_FILES  ${SRC_FILES} ${PATCH_SRC_FILES} PARENT_SCOPE)
