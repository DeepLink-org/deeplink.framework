cmake_minimum_required(VERSION 3.20)
project(TorchDIPU)

option(TESTS "Whether to build unit tests" OFF)
option(LIBS "Whether to build dipu lib, default on" ON)

#require the external project to build as target
include(ExternalProject)

# use gcover
option(ENABLE_COVERAGE "Use gcov" OFF)
message(STATUS ENABLE_COVERAGE=${ENABLE_COVERAGE})

# device related
set(DEVICE
    "camb"
    CACHE STRING "device string, default camb")
list(APPEND DEVICE_CAMB "CAMB" "camb")
list(APPEND DEVICE_CUDA "CUDA" "cuda")
list(APPEND DEVICE_ASCEND "ASCEND" "ascend")
list(APPEND DEVICE_TOPSRIDER "TOPS" "tops" "TOPSRIDER" "topsrider")
list(APPEND DEVICE_SUPA "SUPA" "supa")
list(APPEND DEVICE_DROPLET "DROPLET" "droplet")

execute_process(COMMAND git rev-parse --short HEAD
                OUTPUT_VARIABLE DIPU_GIT_HASH)

set(UsedVendor "")
set(Build_DIOPI_Target "torch")
if (${DEVICE} IN_LIST DEVICE_CUDA)
    set(USE_CUDA ON)
    set(UsedVendor cuda)
elseif (${DEVICE} IN_LIST DEVICE_CAMB)
    set(USE_CAMB ON)
    set(UsedVendor camb)
    set(Build_DIOPI_Target "camb")
elseif (${DEVICE} IN_LIST DEVICE_ASCEND)
    set(USE_ASCEND ON)
    set(UsedVendor ascend)
elseif (${DEVICE} IN_LIST DEVICE_TOPSRIDER)
    set(USE_TOPSRIDER ON)
    set(UsedVendor topsrider)
elseif (${DEVICE} IN_LIST DEVICE_SUPA)
    set(USE_SUPA ON)
    set(UsedVendor supa)
elseif (${DEVICE} IN_LIST DEVICE_DROPLET)
    set(USE_DROPLET ON)
    set(UsedVendor droplet)
else()
    message(FATAL_ERROR "No implementation module is compiled, cmake requires option -DDEVICE=CAMB or CUDA or ASCEND or SUPA")
endif()



string(REGEX MATCH "[a-z0-9_]+" DIPU_GIT_HASH ${DIPU_GIT_HASH})
execute_process(COMMAND sh -c "git status --porcelain | egrep '^(M| M)' | wc -l"
                OUTPUT_VARIABLE DIPU_MODIFY_LEN)
if(DIPU_MODIFY_LEN GREATER 0)
  set(DIPU_GIT_HASH ${DIPU_GIT_HASH}-dirty)
endif()
message(STATUS "DIPU_GIT_HASH: " ${DIPU_GIT_HASH})
add_compile_options(-DDIPU_GIT_HASH="${DIPU_GIT_HASH}")

# Automatically generate a list of supported diopi functions
execute_process(
  COMMAND
    sh -x -c
    "grep -Po 'diopi[a-zA-Z0-9]+(?=\\()' ${CMAKE_CURRENT_SOURCE_DIR}/scripts/autogen_diopi_wrapper/diopi_functions.yaml | sort -u > ${CMAKE_CURRENT_SOURCE_DIR}/SupportedDiopiFunctions.txt"
)

if(NOT DEFINED DIPU_ABI_V)
    execute_process(
      COMMAND
        sh -x -c
        "python -c 'import torch, builtins; print(next(item[-4:-2] for item in dir(builtins)      \
            if \"__pybind11_internals_v4_gcc_libstdcpp_cxxabi10\" in item))'"
      OUTPUT_VARIABLE DIPU_ABI_V)
endif()

if(NOT DEFINED DIPU_COMPILED_WITH_CXX11_ABI)    
    execute_process(
      COMMAND
        sh -x -c
        "python -c 'import torch;print(1 if torch.compiled_with_cxx11_abi() else 0)'"
      OUTPUT_VARIABLE DIPU_COMPILED_WITH_CXX11_ABI)
endif()
    
if(DIPU_COMPILED_WITH_CXX11_ABI GREATER 0)
  set(DIPU_COMPILED_WITH_CXX11_ABI 1)
else()
  set(DIPU_COMPILED_WITH_CXX11_ABI 0)
endif()

if(NOT DEFINED DIOPI_CMAKE_PREFIX_PATH)    
    execute_process(
      COMMAND
        sh -x -c
        "python -c 'import torch;print(torch.utils.cmake_prefix_path)'"
      OUTPUT_VARIABLE DIOPI_CMAKE_PREFIX_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

add_definitions(-fabi-version=${DIPU_ABI_V})
message(STATUS "DIPU_ABI_V: ${DIPU_ABI_V}")

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI}"
)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI})
message(STATUS "DIPU_COMPILED_WITH_CXX11_ABI:" ${DIPU_COMPILED_WITH_CXX11_ABI})

execute_process(
  COMMAND sh -c "dirname $(find  $(dirname $(which python))/../ -name Python.h)"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
message(STATUS "PYTHON_INCLUDE_DIR: " ${PYTHON_INCLUDE_DIR})

if(NOT DEFINED PYTORCH_DIR)
    execute_process(
      COMMAND sh -c "dirname $(python -c 'import torch;print(torch.__path__[0])')"
      OUTPUT_VARIABLE PYTORCH_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

message(STATUS "PYTORCH_DIR: " ${PYTORCH_DIR})

# config
include(cmake/BaseFuncions.cmake)
_set_cpp_flags()


if(ENABLE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# TODO(lljbash): findTorch encounter some error, need check
# ~~~
# set(Torch_DIR ${PYTORCH_DIR}/share/cmake/Torch)
# find_package(Torch REQUIRED)
# if (NOT Torch_FOUND)
#     message(FATAL_ERROR "libtorch is required but not found")
# else()
#     link_directories(${PYTORCH_DIR}/torch/lib)
#     include_directories(${TORCH_INCLUDE_DIRS})
#     message(STATUS "Found Torch Version: ${Torch_VERSION}")
#     message(STATUS "Torch TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
# endif()
# ~~~

set(TORCH_LIBRARY_DIR "${PYTORCH_DIR}/torch/lib")
link_directories(${TORCH_LIBRARY_DIR})
list(APPEND TORCH_INCLUDE_DIRS ${PYTORCH_DIR}/torch/include/
     ${PYTORCH_DIR}/torch/include/torch/csrc/api/include/)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

add_library(diopi_impl SHARED IMPORTED)

if(WITH_DIOPI STREQUAL "INTERNAL")
#----------------------------------Build DIOPI submodule-------------------------------------------
  set(DIOPI_INTERNAL diopi_internal)
  set(DIOPI_SUBDIR impl)
  set(DIOPI_BUILD_TARGET "${DIOPI_INTERNAL}-build")
  set(DIOPI_DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third_party")
  set(DIOPI_SRC_PATH "${PROJECT_SOURCE_DIR}/third_party/DIOPI")
  set(DIOPI_BUILD_PATH "${DIOPI_SRC_PATH}/build")
  set(DIOPI_PROTO_PATH "${DIOPI_SRC_PATH}/proto")
  set(DIOPI_IMPL_LIB_PATH "${DIOPI_SRC_PATH}/impl/lib")
  set(DIOPI_IMPL_LIB "${DIOPI_IMPL_LIB_PATH}/libdiopi_impl.so")
  ExternalProject_Add(${DIOPI_INTERNAL}
                      DOWNLOAD_DIR ${DIOPI_DOWNLOAD_PATH}
                      SOURCE_DIR ${DIOPI_SRC_PATH}
                      SOURCE_SUBDIR ${DIOPI_SUBDIR}
                      BINARY_DIR ${DIOPI_BUILD_PATH}
                      DOWNLOAD_COMMAND ""
                      CMAKE_ARGS -DIMPL_OPT=${Build_DIOPI_Target}
                                 -DENABLE_COVERAGE=${USE_COVERAGE} 
                                 -DCMAKE_PREFIX_PATH=${DIOPI_CMAKE_PREFIX_PATH}
                      BUILD_BYPRODUCTS ${DIOPI_IMPL_LIB}
                      INSTALL_COMMAND cmake -E echo "Skipping install step for ${DIOPI_INTERNAL}."
  )
  ExternalProject_Add_StepTargets(${DIOPI_INTERNAL} configure build)
  ExternalProject_Add_StepDependencies(${DIOPI_INTERNAL} build ${DIOPI_INTERNAL}-configure)
  add_dependencies(diopi_impl ${DIOPI_BUILD_TARGET})
#--------------------------------------------------------------------------------------------------
else()
    if(EXISTS $ENV{DIOPI_PATH})
      set(DIOPI_PROTO_PATH $ENV{DIOPI_PATH})
    else()
      set(DIOPI_PROTO_PATH "${WITH_DIOPI}/proto")
    endif()
    if(EXISTS $ENV{DIOPI_ROOT})
      set(DIOPI_IMPL_LIB_PATH $ENV{DIOPI_ROOT})
      set(DIOPI_IMPL_LIB "${DIOPI_IMPL_LIB_PATH}/libdiopi_impl.so")
    else()
      set(DIOPI_IMPL_LIB_PATH "${WITH_DIOPI}/impl/lib")
      set(DIOPI_IMPL_LIB "${DIOPI_IMPL_LIB_PATH}/libdiopi_impl.so")
    endif()
endif()

message(STATUS "DIOPI_PROTO_PATH: ${DIOPI_PROTO_PATH}")
message(STATUS "DIOPI_IMPL_LIB_PATH: ${DIOPI_IMPL_LIB_PATH}")

set_target_properties(
  diopi_impl PROPERTIES IMPORTED_LOCATION
                        ${DIOPI_IMPL_LIB})
target_include_directories(diopi_impl INTERFACE "${DIOPI_PROTO_PATH}/include")
target_compile_definitions(diopi_impl INTERFACE DIOPI_ATTR_WEAK)

if(LIBS)
  add_subdirectory(torch_dipu/csrc_dipu)
endif()

if(TESTS)
  add_subdirectory(tests/cpp)
endif()
