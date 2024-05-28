# CMake script to locate XCCL

include(FindPackageHandleStandardArgs)

set(_NCCL "NCCL")
set(_MCCL "MCCL")
set(_XCCL_NAME ${NAMES})
if(NOT NAMES)
    set(_XCCL_NAME ${_NCCL})
endif()

if (${_XCCL_NAME} strequal ${_NCCL})

    find_path(XCCL_INCLUDE_DIR
          NAMES nccl.h
          HINTS $ENV{CUCC_PATH}/include)
    find_library(XCCL_LIBRARIES
          NAMES mccl
          HINTS $ENV{MACA_PATH}/lib)
else()

    find_path(XCCL_INCLUDE_DIR
          NAMES nccl.h
          HINTS ${NCCL_ROOT}/include
          $ENV{NCCL_ROOT}/include
          /usr/include
          /usr/local/include)
    find_library(XCCL_LIBRARIES
          NAMES nccl
          HINTS ${NCCL_ROOT}/lib
          $ENV{NCCL_ROOT}/lib
          /usr/lib
          /usr/local/lib)
endif()


if (XCCL_INCLUDE_DIR)
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
endif (XCCL_INCLUDE_DIR)

if(NOT NCCL_VERSION_MAJOR)
    set(XCCL_FOUND FALSE)
    set(XCCL_VERSION "???")
else()
    set(XCCL_FOUND TRUE)
    set(XCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
endif()

find_package_handle_standard_args(XCCL
    REQUIRED_VARS XCCL_INCLUDE_DIR XCCL_LIBRARIES
    VERSION_VAR XCCL_VERSION)
