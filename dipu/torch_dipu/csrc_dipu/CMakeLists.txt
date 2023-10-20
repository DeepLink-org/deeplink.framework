set(DIPU_LIB torch_dipu)
set(DIPU_PYTHON_LIB torch_dipu_python)
set(DIPU_AUTOGENED_KERNELS torch_dipu_autogened_kernels)

# python path
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

# dipu include path
set(DIPU_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${DIPU_PATH})

set(DIST_DIR ${DIPU_PATH}/dist/include)
include_directories(${DIST_DIR})

# src
file(
  GLOB
  RT_SRC_FILES
  runtime/core/guardimpl/*.cpp
  runtime/core/allocator/*.cpp
  runtime/core/DIPU*.cpp
  runtime/core/MemChecker.cpp
  runtime/distributed/*.cpp
  runtime/devproxy/*.cpp)

file(
  GLOB
  OP_SRC_FILES
  aten/RegisterDIPU.cpp
  aten/CPUFallback.cpp
  aten/util/*.cpp
  aten/ops/*Kernel*.cpp
  aten/ops/DIPUAmp.cpp
  aten/ops/CustomFallbackFunctionsForAmpGradScaler.cpp)

file(GLOB BASE_FILES base/*.cpp)
file(GLOB UTILS_FILES utils/*.cpp)
file(GLOB DIOPI_RT_FILES diopirt/*.cpp)
file(GLOB PROFILER_FILES profiler/*.cpp)

# vendor src
add_subdirectory(vendor/${UsedVendor})
include_directories(SYSTEM ${VENDOR_INCLUDE_DIRS})
link_directories(${VENDOR_LIB_DIRS})

set(SOURCE_FILES
    ${RT_SRC_FILES}
    ${OP_SRC_FILES}
    ${DIOPI_RT_FILES}
    ${BASE_FILES}
    ${UTILS_FILES}
    ${VENDOR_FILES}
    ${PROFILER_FILES})

add_library(${DIPU_LIB} SHARED ${SOURCE_FILES})

# link
target_link_libraries(${DIPU_LIB} ${DIPU_VENDOR_LIB})
target_link_libraries(${DIPU_LIB} -Wl,--no-as-needed diopi_impl -Wl,--as-needed)
target_link_libraries(${DIPU_LIB} c10 torch torch_cpu)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${DIPU_LIB} Threads::Threads)

# copy vendor header file
set(VENDOR_DIST "${DIST_DIR}/csrc_dipu/vendor/")
set(VENDOR_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/${UsedVendor}")
add_custom_command(
  OUTPUT vendor_include
  COMMAND mkdir -p ${VENDOR_DIST}
  COMMAND cmake -E create_symlink ${VENDOR_HEADER_DIR}/vendorapi.h
          ${VENDOR_DIST}/vendorapi.h)
if(EXISTS "${VENDOR_HEADER_DIR}/vendor_autocast.h"
   AND NOT IS_DIRECTORY "${VENDOR_HEADER_DIR}/vendor_autocast.h")
  add_custom_command(
    OUTPUT vendor_include
    APPEND
    COMMAND cmake -E create_symlink ${VENDOR_HEADER_DIR}/vendor_autocast.h
            ${VENDOR_DIST}/vendor_autocast.h)
else()
  target_compile_definitions(${DIPU_LIB} PRIVATE DIPU_NO_VENDOR_AUTOCAST)
endif()
add_custom_target(copy_include DEPENDS vendor_include)
add_dependencies(${DIPU_LIB} copy_include)

add_custom_target(${DIPU_AUTOGENED_KERNELS})
add_dependencies(${DIPU_LIB} ${DIPU_AUTOGENED_KERNELS})

# ---------------------- autogen kernel codes ----------------------------------
set(DIPU_AUTOGEN_DIOPI_WRAPPER_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/scripts/autogen_diopi_wrapper")
set(DIPU_AUTOGEN_DIOPI_WRAPPER_SCRIPT
    "${DIPU_AUTOGEN_DIOPI_WRAPPER_SOURCE_DIR}/autogen_diopi_wrapper.py")
set(DIPU_AUTOGEN_DIOPI_WRAPPER_CONFIG
    "${DIPU_AUTOGEN_DIOPI_WRAPPER_SOURCE_DIR}/diopi_functions.yaml")
set(DIPU_AUTOGENED_KERNELS_CPP
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/AutoGenedKernels.cpp")
add_custom_command(
  OUTPUT "${DIPU_AUTOGENED_KERNELS_CPP}"
  COMMAND
    python "${DIPU_AUTOGEN_DIOPI_WRAPPER_SCRIPT}" --config
    "${DIPU_AUTOGEN_DIOPI_WRAPPER_CONFIG}" --out "${DIPU_AUTOGENED_KERNELS_CPP}"
    --use_diopi_adapter False --autocompare False --print_func_call_info True
    --print_op_arg True --fun_config_dict
    '{\"current_device\": \"${UsedVendor}\"}'
  DEPENDS ${DIPU_AUTOGEN_DIOPI_WRAPPER_SCRIPT}
          ${DIPU_AUTOGEN_DIOPI_WRAPPER_CONFIG})
add_custom_target(autogen_diopi_kernels_cpp
                  DEPENDS ${DIPU_AUTOGENED_KERNELS_CPP})
add_dependencies(${DIPU_AUTOGENED_KERNELS} autogen_diopi_kernels_cpp)

# custom diopi functions (now only for camb)
if(${UsedVendor} STREQUAL "camb")
  set(DIPU_AUTOGEN_DIOPI_WRAPPER_CUSTOM_CONFIG
      "${DIPU_AUTOGEN_DIOPI_WRAPPER_SOURCE_DIR}/custom_diopi_functions.yaml")
  set(DIPU_CUSTOM_AUTOGENED_KERNELS_CPP
      "${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/CustomAutoGenedKernels.cpp")
  add_custom_command(
    OUTPUT ${DIPU_CUSTOM_AUTOGENED_KERNELS_CPP}
    COMMAND
      python ${DIPU_AUTOGEN_DIOPI_WRAPPER_SCRIPT} --config
      ${DIPU_AUTOGEN_DIOPI_WRAPPER_CUSTOM_CONFIG} --out
      ${DIPU_CUSTOM_AUTOGENED_KERNELS_CPP} --use_diopi_adapter False
    DEPENDS ${DIPU_AUTOGEN_DIOPI_WRAPPER_SCRIPT}
            ${DIPU_AUTOGEN_DIOPI_WRAPPER_CUSTOM_CONFIG})
  add_custom_target(autogen_diopi_custom_kernels_cpp
                    DEPENDS ${DIPU_CUSTOM_AUTOGENED_KERNELS_CPP})
  add_dependencies(${DIPU_AUTOGENED_KERNELS} autogen_diopi_custom_kernels_cpp)
endif()
# ------------------------------------------------------------------------------

# ---------------------- build bind in python ----------------------------------
file(GLOB BIND_SRC_FILES binding/Export*.cpp binding/patch*.cpp)
set(BIND_FILES ${BIND_SRC_FILES})
add_library(${DIPU_PYTHON_LIB} SHARED ${BIND_SRC_FILES})
# TODO: default hidden setting scope is incorrect and cannot open now because it
# cause diopirt hidden, so temporarily use this target level setting. enhance in
# future.
set_target_properties(${DIPU_PYTHON_LIB} PROPERTIES CXX_VISIBILITY_PRESET
                                                    hidden)
target_link_libraries(${DIPU_PYTHON_LIB} ${DIPU_LIB})
# ------------------------------------------------------------------------------

# -------------------- build _C.python extension -------------------------------
execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE DIPU_C_PYTHON_EXT_ARCH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(PythonInterp REQUIRED)

string(CONCAT DIPU_C_PYTHON_EXT_LIB
              "_C.cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-"
              "${DIPU_C_PYTHON_EXT_ARCH}-linux-gnu")
add_library(${DIPU_C_PYTHON_EXT_LIB} SHARED stub.cpp)
set_target_properties(
  ${DIPU_C_PYTHON_EXT_LIB}
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/torch_dipu PREFIX
                                                                       "")
target_link_libraries(${DIPU_C_PYTHON_EXT_LIB} ${DIPU_PYTHON_LIB} torch_python)
target_compile_options(${DIPU_C_PYTHON_EXT_LIB} PRIVATE -fstack-protector-all)
# -----------------------------------------------------------------------------
