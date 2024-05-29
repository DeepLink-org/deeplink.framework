# CMake script to locate XCCL, use simple Module mode.

include(FindPackageHandleStandardArgs)

set(_NCCL "NCCL")
set(_MCCL "MCCL")
if(NOT _XCCL_NAME)
    set(_XCCL_NAME ${_NCCL})
endif()

if (${_XCCL_NAME} STREQUAL ${_NCCL})
    find_path(XCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${NCCL_ROOT}/include
        $ENV{NCCL_ROOT}/include
        /usr/include
        /usr/local/include)
    find_library(XCCL_LIB
        NAMES nccl
        HINTS ${NCCL_ROOT}/lib
        $ENV{NCCL_ROOT}/lib
        /usr/lib
        /usr/local/lib)
else()
    find_path(XCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${CUCC_PATH}/include
        HINTS $ENV{CUCC_PATH}/include)
    find_library(XCCL_LIB
        NAMES mccl
        HINTS ${MACA_PATH}/lib
        HINTS $ENV{MACA_PATH}/lib)
endif()
get_filename_component(XCCL_LIB_DIR ${XCCL_LIB} DIRECTORY)

if (XCCL_INCLUDE_DIR)
    if (${_XCCL_NAME} STREQUAL ${_NCCL})
        file(READ ${XCCL_INCLUDE_DIR}/nccl.h NCCL_VERSION_FILE_CONTENTS)
        string(REGEX MATCH "define NCCL_MAJOR * +([0-9]+)"
                NCCL_VERSION_MAJOR "${NCCL_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define NCCL_MAJOR * +([0-9]+)" "\\1"
                NCCL_VERSION_MAJOR "${NCCL_VERSION_MAJOR}")
        string(REGEX MATCH "define NCCL_MINOR * +([0-9]+)"
                NCCL_VERSION_MINOR "${NCCL_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define NCCL_MINOR * +([0-9]+)" "\\1"
                NCCL_VERSION_MINOR "${NCCL_VERSION_MINOR}")
        string(REGEX MATCH "define NCCL_PATCH * +([0-9]+)"
                NCCL_VERSION_PATCH "${NCCL_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define NCCL_PATCH * +([0-9]+)" "\\1"
                NCCL_VERSION_PATCH "${NCCL_VERSION_PATCH}")
        set(XCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    else()
        set(XCCL_VERSION "muxi-mccl")
    endif()
endif()

find_package_handle_standard_args(XCCL
    REQUIRED_VARS XCCL_INCLUDE_DIR XCCL_LIB_DIR
    VERSION_VAR XCCL_VERSION)
