set(XPURT_TOOLKIT_ROOT /workspace/xpu_toolchain/xpurt)
set(XDNN_TOOLKIT_ROOT /workspace/xpu_toolchain/xdnn)
set(XCCL_TOOLKIT_ROOT /workspace/xpu_toolchain/xccl)

include(FindPackageHandleStandardArgs)

## xdnn
find_path(XDNN_INCLUDE_DIR
    NAMES xpu/xdnn.h
    HINTS ${XDNN_TOOLKIT_ROOT}/include
          $ENV{XDNN_TOOLKIT_ROOT}/include
)
message("XDNN_INCLUDE_DIR:" ${XDNN_INCLUDE_DIR})
find_library(XDNN_LIBRARIES
    NAMES xpuapi
    HINTS ${XDNN_TOOLKIT_ROOT}/so
          $ENV{XDNN_TOOLKIT_ROOT}/so
)
message("XDNN_TOOLKIT_ROOT: " ${XDNN_TOOLKIT_ROOT})
message("XDNN_LIBRARIES:" ${XDNN_LIBRARIES})
if(NOT XDNN_INCLUDE_DIR OR NOT XDNN_LIBRARIES)
    message(FATAL_ERROR "Cannot find Xdnn TOOLKIT for kunlunxin, set ENV 'XDNN_TOOLKIT_ROOT' correctly")
endif()

## runtime
find_path(XPURT_INCLUDE_DIR
    NAMES xpu/runtime.h
    HINTS ${XPURT_TOOLKIT_ROOT}/include
          $ENV{XPURT_TOOLKIT_ROOT}/include
)
message("XPURT_INCLUDE_DIR:" ${XPURT_INCLUDE_DIR})
find_library(XPURT_LIBRARIES
    NAMES xpurt
    HINTS ${XPURT_TOOLKIT_ROOT}/so
          $ENV{XPURT_TOOLKIT_ROOT}/so
)
message("XPURT_LIBRARIES:" ${XPURT_LIBRARIES})
if(NOT XPURT_INCLUDE_DIR OR NOT XPURT_LIBRARIES)
    message(FATAL_ERROR "Cannot find XPURT TOOLKIT for kunlunxin, set ENV 'XPURT_TOOLKIT_ROOT' correctly")
endif()

## xccl
find_path(XCCL_INCLUDE_DIR
    NAMES bkcl.h
    HINTS ${XCCL_TOOLKIT_ROOT}/include
          $ENV{XCCL_TOOLKIT_ROOT}/include
)
message("XCCL_INCLUDE_DIR:" ${XCCL_INCLUDE_DIR})
find_library(XCCL_LIBRARIES
    NAMES bkcl
    HINTS ${XCCL_TOOLKIT_ROOT}/so
          $ENV{XCCL_TOOLKIT_ROOT}/so
)
message("XCCL_LIBRARIES:" ${XCCL_LIBRARIES})
if(NOT XCCL_INCLUDE_DIR OR NOT XCCL_LIBRARIES)
    message(FATAL_ERROR "Cannot find XCCL TOOLKIT for kunlunxin, set ENV 'XCCL_TOOLKIT_ROOT' correctly")
endif()

find_package_handle_standard_args(XPURT DEFAULT_MSG
    XPURT_INCLUDE_DIR
    XPURT_LIBRARIES)

find_package_handle_standard_args(XDNN DEFAULT_MSG
    XDNN_INCLUDE_DIR
    XDNN_LIBRARIES)

find_package_handle_standard_args(XCCL DEFAULT_MSG
    XCCL_INCLUDE_DIR
    XCCL_LIBRARIES)

mark_as_advanced(XPURT_INCLUDE_DIR XPURT_LIBRARIES XDNN_INCLUDE_DIR XDNN_LIBRARIES XCCL_INCLUDE_DIR XCCL_LIBRARIES)
