find_path(NEUWARE_ROOT_DIR
    NAMES include/cnrt.h
    HINTS ${NEUWARE_ROOT}
          $ENV{NEUWARE_ROOT}
          ${CAMB_PATH}
          $ENV{CAMB_PATH}
          /usr/local/neuware
)

find_program(NEUWARE_CNCC_EXECUTABLE
    NAMES cncc
    HINTS ${NEUWARE_ROOT_DIR}/bin
          ${NEUWARE_ROOT}/bin
          $ENV{NEUWARE_ROOT}/bin
          ${CAMB_PATH}/bin
          $ENV{CAMB_PATH}/bin
          /usr/local/neuware/bin
)


if(NOT NEUWARE_ROOT_DIR)
    message(FATAL_ERROR "Cannot find NEUWARE SDK for cambricon, set ENV 'NEUWARE_ROOT' correctly")
endif()

# get CNRT version
file(READ ${NEUWARE_ROOT_DIR}/include/cnrt.h CNRT_FILE_CONTENTS)
string(REGEX MATCH "define CNRT_MAJOR_VERSION * +([0-9]+)"
    CNRT_MAJOR_VERSION "${CNRT_FILE_CONTENTS}")
string(REGEX REPLACE "define CNRT_MAJOR_VERSION * +([0-9]+)" "\\1"
    CNRT_MAJOR_VERSION "${CNRT_MAJOR_VERSION}")
string(REGEX MATCH "define CNRT_MINOR_VERSION * +([0-9]+)"
    CNRT_MINOR_VERSION "${CNRT_FILE_CONTENTS}")
string(REGEX REPLACE "define CNRT_MINOR_VERSION * +([0-9]+)" "\\1"
    CNRT_MINOR_VERSION "${CNRT_MINOR_VERSION}")
string(REGEX MATCH "define CNRT_PATCH_VERSION * +([0-9]+)"
    CNRT_PATCH_VERSION "${CNRT_FILE_CONTENTS}")
string(REGEX REPLACE "define CNRT_PATCH_VERSION * +([0-9]+)" "\\1"
    CNRT_PATCH_VERSION "${CNRT_PATCH_VERSION}")

set(CNRT_VERSION "${CNRT_MAJOR_VERSION}.${CNRT_MINOR_VERSION}.${CNRT_PATCH_VERSION}")

set(CNCC_TARGETS $ENV{CNCC_TARGETS})
if ((NOT CNCC_TARGETS) OR "x${CNCC_TARGETS}" STREQUAL "x")
    set(CNCC_TARGETS "mtp_290" "mtp_372")
endif()

set(CMAKE_CNCC_FLAGS "-I${NEUWARE_ROOT_DIR}/include" "--std=c++11")
if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND CMAKE_CNCC_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
else()
    set(CMAKE_CNCC_FLAGS "${CMAKE_CNCC_FLAGS};-O3")
endif()

set(BANG_CNCC_FLAGS)
foreach(target ${CNCC_TARGETS})
    list(APPEND BANG_CNCC_FLAGS "--bang-mlu-arch=${target}")
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Neuware
  REQUIRED_VARS
    NEUWARE_ROOT_DIR
    NEUWARE_CNCC_EXECUTABLE
    CMAKE_CNCC_FLAGS
    CNCC_TARGETS
  VERSION_VAR
    CNRT_VERSION
)


# setup flags for C++ compile
include_directories(SYSTEM "${NEUWARE_ROOT_DIR}/include")
link_directories("${NEUWARE_ROOT_DIR}/lib64")

macro(CAMB_COMPILE generated_files)
    foreach(src_file ${ARGN})
        get_filename_component(basename ${src_file} NAME)
        set(output_file_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${basename}_${target}_target.dir")
        set(compiled_file "${output_file_dir}/${basename}.${target}.o")

        add_custom_command(
            OUTPUT ${compiled_file}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${output_file_dir}"
            COMMAND ${NEUWARE_CNCC_EXECUTABLE} ${CMAKE_CNCC_FLAGS}
                    ${BANG_CNCC_FLAGS} "-o" ${compiled_file}
                    "-c" ${src_file}
            WORKING_DIRECTORY ${output_file_dir}
        )
        add_custom_target(${basename}_${target}_target DEPENDS ${compiled_file})
        list(APPEND ${generated_files} ${compiled_file})
    endforeach()
endmacro()
