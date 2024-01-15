cmake_minimum_required(VERSION 3.14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

include(FindKLXRuntime)

set(VENDOR_INCLUDE_DIRS ${VENDOR_INCLUDE_DIRS} ${XPURT_INCLUDE_DIR} ${XDNN_INCLUDE_DIR} ${XCCL_INCLUDE_DIR} PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${VENDOR_LIB_DIRS} ${XPURT_LIBRARIES} ${XDNN_LIBRARIES} ${XCCL_LIBRARIES} PARENT_SCOPE)
set(VENDOR_INCLUDE_DIRS ${VENDOR_INCLUDE_DIRS} /workspace/xpu_toolchain/xpurt/include /workspace/xpu_toolchain/xdnn/include /workspace/xpu_toolchain/xccl/include PARENT_SCOPE)
set(VENDOR_LIB_DIRS /workspace/xpu_toolchain/xdnn/so/ /workspace/xpu_toolchain/xpurt/so/ /workspace/xpu_toolchain/xccl/so/ PARENT_SCOPE)

set(DIPU_VENDOR_LIB ${DIPU_VENDOR_LIB} xpurt xpuapi bkcl PARENT_SCOPE)

file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)
