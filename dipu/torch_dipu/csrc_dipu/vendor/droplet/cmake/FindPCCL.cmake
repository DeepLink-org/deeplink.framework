find_path(
    PCCL_INCLUDE_DIR pccl.h
    HINTS $ENV{PCCL_PATH} ${PCCL_PATH} /usr/local/pccl
    PATH_SUFFIXES include
)

find_library(
    PCCL_LIBRARY pccl
    HINTS $ENV{PCCL_PATH} ${PCCL_PATH} /usr/local/pccl
    PATH_SUFFIXES lib64 lib/x64 lib lib/linux-x86_64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    PCCL
    REQUIRED_VARS
    PCCL_INCLUDE_DIR PCCL_LIBRARY
)
