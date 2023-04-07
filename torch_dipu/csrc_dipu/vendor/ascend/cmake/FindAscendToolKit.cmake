find_path(ASCEND_TOOLKIT_ROOT
    NAMES include/acl/acl.h
    HINTS ${ASCEND_TOOLKIT_ROOT}
          $ENV{ASCEND_TOOLKIT_ROOT}
          ${ASCEND_PATH}
          $ENV{ASCEND_PATH}
        /usr/local/Ascend/ascend-toolkit/latest/
)


if(NOT ASCEND_TOOLKIT_ROOT)
    message(FATAL_ERROR "Cannot find SDK for ascend, set ENV 'ASCEND_TOOLKIT_ROOT' correctly")
endif()


# setup flags for C++ compile
include_directories(SYSTEM "${ASCEND_TOOLKIT_ROOT}/include")
link_directories("${ASCEND_TOOLKIT_ROOT}/lib64")
