cmake_minimum_required(VERSION 3.5)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


set(DROPLET_INSTALL "/usr/local/tangrt")
include(cmake/FindDropletCompiler.cmake)

set(VENDOR_INCLUDE_DIRS "${DROPLET_INSTALL}/include" PARENT_SCOPE)
set(VENDOR_LIB_DIRS "${DROPLET_INSTALL}/lib/linux-x86_64" PARENT_SCOPE)
set(DIPU_VENDOR_LIB tangrt_shared PARENT_SCOPE)


file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)
