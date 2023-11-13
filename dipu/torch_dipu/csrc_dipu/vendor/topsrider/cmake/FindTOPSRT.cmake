include(FindPackageHandleStandardArgs)

find_path(TOPSRT_INCLUDE_DIR
    NAMES tops_runtime.h
    HINTS /usr/include/tops/ /usr/include/ /usr/include/dtu/ /usr/include/dtu/tops
    /usr/include/dtu/3_0/runtime
    /home/cse/src/install/usr/include/tops
    /opt/tops/include/
    /opt/tops/include/tops/
)

find_path(TOPSRT_LIBRARIES_DIR
    NAMES libtopsrt.so
    HINTS /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /home/cse/src/install/usr/lib
    /opt/tops/lib/
)

find_package_handle_standard_args(TOPSRT DEFAULT_MSG
    TOPSRT_INCLUDE_DIR
    TOPSRT_LIBRARIES_DIR)

mark_as_advanced(TOPSRT_INCLUDE_DIR TOPSRT_LIBRARIES_DIR)
