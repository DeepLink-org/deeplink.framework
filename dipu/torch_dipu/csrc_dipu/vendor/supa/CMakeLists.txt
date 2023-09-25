set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})


if(NOT DEFINED ENV{SUPA_PATH})
    set(SUPA_PATH /usr/local/supa)
else()
    set(SUPA_PATH $ENV{SUPA_PATH})
endif()

find_path(SUPA_INCLUDE_DIR supa.h PATHS
  ${SUPA_PATH}/include
)

IF (NOT SUPA_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR " Could not find supa.h. Make sure fullstack is installed, or SUPA_PATH is set properly.")
ENDIF()

find_library(SUPA_SUPART_LIBRARY supa-runtime
  ${SUPA_PATH}/lib ${SUPA_PATH}/lib64
)
IF (NOT SUPA_SUPART_LIBRARY)
  MESSAGE(FATAL_ERROR " Could not find supa-runtime lib. Make sure fullstack is installed, or SUPA_PATH is set properly.")
ENDIF()

find_library(SUPA_SUCCL_LIBRARY succl
  ${SUPA_PATH}/lib ${SUPA_PATH}/lib64 /usr/lib/x86_64-linux-gnu
)
IF (NOT SUPA_SUCCL_LIBRARY)
  MESSAGE(FATAL_ERROR " Could not find supa succl lib. Make sure fullstack is installed, or SUPA_PATH is set properly.")
ENDIF()

# it's incorrect, need enhance find cmake to set lib fullpath
# set(VENDOR_INCLUDE_DIRS "${ASCEND_TOOLKIT_ROOT}/include" PARENT_SCOPE)
# set(VENDOR_LIB_DIRS "${ASCEND_TOOLKIT_ROOT}/lib64" PARENT_SCOPE)
# set(DIPU_VENDOR_LIB ascendcl acl_op_compiler PARENT_SCOPE)

file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)

set(VENDOR_INCLUDE_DIRS ${SUPA_PATH}/include PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${SUPA_PATH}/lib ${SUPA_PATH}/lib64 PARENT_SCOPE)
set(DIPU_VENDOR_LIB ${SUPA_SUPART_LIBRARY} ${SUPA_SUCCL_LIBRARY} PARENT_SCOPE)
