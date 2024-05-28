find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
    message(STATUS "CUDA include: " ${CUDA_INCLUDE_DIRS})
    message(STATUS "CUCC Path: ${CUDA_TOOLKIT_ROOT_DIR}/bin")
    set(SRC_FILES
        ../cudapatch/DIPUPatchCudaAllocator.cpp
        ../cudapatch/wrapperRegister.cpp)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/cmake)
find_package(XCCL  NAMES "MCCL"  REQUIRED)

message(STATUS "MCCL include Path:  ${XCCL_INCLUDE_DIR}")
message(STATUS "MCCL lib Path:  ${XCCL_LIBRARIES}")

set(VENDOR_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${XCCL_INCLUDE_DIR} PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${XCCL_LIBRARIES} PARENT_SCOPE)


set(SRC_FILES
    ../cuda/communiatorimpl.cpp
    ../cuda/CUDACopyInplace.cpp
    ../cuda/CudaGeneratorImpl.cpp
    ../cuda/deviceimpl.cpp)

set(VENDOR_FILES  ${SRC_FILES} ${PATCH_SRC_FILES} PARENT_SCOPE)
