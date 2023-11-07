cmake_minimum_required(VERSION 3.5)

option(DIST "whether to use PCCL to implement communicator" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(DROPLET_INSTALL "/usr/local/tangrt")
include(cmake/FindDropletCompiler.cmake)

if (DIST)
    message(STATUS "use PCCL to implement communicator")
    find_package(PCCL REQUIRED)
    message(STATUS "PCCL_LIBRARY: " ${PCCL_LIBRARY})
    message(STATUS "PCCL_INCLUDE_DIR: " ${PCCL_INCLUDE_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PCCL" PARENT_SCOPE)
    set(DIPU_VENDOR_LIB ${DIPU_VENDOR_LIB} ${PCCL_LIBRARY})
    set(VENDOR_INCLUDE_DIRS ${VENDOR_INCLUDE_DIRS} ${PCCL_INCLUDE_DIR})

else()
    message(STATUS "do not use PCCL to implement communicator, i.e. not support distributed on droplet")
endif()


set(VENDOR_INCLUDE_DIRS ${VENDOR_INCLUDE_DIRS} ${DROPLET_INSTALL}/include PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${VENDOR_LIB_DIRS} ${DROPLET_INSTALL}/lib/linux-x86_64 PARENT_SCOPE)
set(DIPU_VENDOR_LIB ${DIPU_VENDOR_LIB} tangrt_shared PARENT_SCOPE)

file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)
