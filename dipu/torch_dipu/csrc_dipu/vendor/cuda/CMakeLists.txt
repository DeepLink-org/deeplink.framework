# if torch compiled with cuda, find torch do find cuda. compile only cpu not.
find_package(CUDA "11.0" REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(XCCL REQUIRED)

if (CUDA_FOUND)
    message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
    message(STATUS "CUDA include: " ${CUDA_INCLUDE_DIRS})
    message(STATUS "NVCC Path: ${CUDA_TOOLKIT_ROOT_DIR}/bin")
    message(STATUS "NCCL include Path:  ${XCCL_INCLUDE_DIR}")
    message(STATUS "NCCL lib Path:  ${XCCL_LIBRARIES}")
endif()
set(VENDOR_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${XCCL_INCLUDE_DIR} PARENT_SCOPE)

set(VENDOR_LIB_DIRS ${XCCL_LIB_DIR} PARENT_SCOPE)
set(DIPU_VENDOR_LIB nccl ${CUDA_LIBRARIES} PARENT_SCOPE)

message(STATUS "DIPU_TORCH_HAS_CUDA = ${DIPU_TORCH_HAS_CUDA}")
if (CUDA_FOUND AND ${DIPU_TORCH_HAS_CUDA})
    # this file depend on c10_cuda. 
    file(GLOB PATCH_SRC_FILES  patch/*.cpp)
endif()

file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} ${PATCH_SRC_FILES} PARENT_SCOPE)
