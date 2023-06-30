cmake_minimum_required(VERSION 3.4)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


set(STPU_INSTALL "/usr/local/tangrt-0.1")
include(cmake/FindStpuCompiler.cmake)

set(VENDOR_INCLUDE_DIRS "${STPU_INSTALL}/include" PARENT_SCOPE)
set(VENDOR_LIB_DIRS "${STPU_INSTALL}/lib/linux-x86_64" PARENT_SCOPE)
set(DIPU_VENDOR_LIB tangrt_shared PARENT_SCOPE)


file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)
