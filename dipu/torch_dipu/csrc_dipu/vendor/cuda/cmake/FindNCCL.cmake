# CMake script to locate NCCL

include(FindPackageHandleStandardArgs)

find_path(NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${NCCL_ROOT}/include
        $ENV{NCCL_ROOT}/include
        /usr/include
        /usr/local/include
        )

find_library(NCCL_LIBRARIES
        NAMES nccl
        HINTS ${NCCL_ROOT}/lib
        $ENV{NCCL_ROOT}/lib
        /usr/lib
        /usr/local/lib
        )

if (NCCL_INCLUDE_DIR)
    file(READ ${NCCL_INCLUDE_DIR}/nccl.h NCCL_VERSION_FILE_CONTENTS)
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
endif (NCCL_INCLUDE_DIR)

if(NOT NCCL_VERSION_MAJOR)
    set(NCCL_FOUND FALSE)
    set(NCCL_VERSION "???")
else()
    set(NCCL_FOUND TRUE)
    set(NCCL_VERSION_STRING "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
endif()

find_package_handle_standard_args(NCCL DEFAULT_MSG
        NCCL_INCLUDE_DIR
        NCCL_LIBRARIES)

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARIES)
