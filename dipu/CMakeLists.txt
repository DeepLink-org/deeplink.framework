cmake_minimum_required(VERSION 3.20)
project(TorchDIPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# the default CMAKE_BUILD_TYPE is Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(ENABLE_LTO "Enable Link Time Optimization in Release build" ON)
if(ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT supported OUTPUT error)
  if(supported)
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    set(CMAKE_POLICY_DEFAULT_CMP0069
        "NEW"
        CACHE STRING "enforce IPO in subprojects" FORCE)
  else()
    message(STATUS "IPO / LTO not supported: <${error}>")
  endif()
else()
  message(STATUS "IPO / LTO disabled")
endif()

option(TESTS "Whether to build unit tests" OFF)
option(LIBS "Whether to build dipu lib, default on" ON)
option(ENABLE_DIPU_EXTRA_FEATURE "Whether to generate extra code for AutoGenedKernels" ON)

# use gcover
option(ENABLE_COVERAGE "Use gcov" OFF)
message(STATUS ENABLE_COVERAGE=${ENABLE_COVERAGE})

# device related
set(DEVICE
    "camb"
    CACHE STRING "device string, default camb")
list(APPEND DEVICE_CAMB "CAMB" "camb")
list(APPEND DEVICE_CUDA "CUDA" "cuda")
list(APPEND DEVICE_MUXI "MUXI" "muxi")
list(APPEND DEVICE_ASCEND "ASCEND" "ascend")
list(APPEND DEVICE_TOPSRIDER "TOPS" "tops" "TOPSRIDER" "topsrider")
list(APPEND DEVICE_SUPA "SUPA" "supa")
list(APPEND DEVICE_DROPLET "DROPLET" "droplet")
list(APPEND DEVICE_KUNLUNXIN "kunlunxin" "klx")

execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE DIPU_GIT_HASH)

set(UsedVendor "")
set(DIOPI_IMPL_OPT "")
if(${DEVICE} IN_LIST DEVICE_CUDA)
  set(USE_CUDA ON)
  set(UsedVendor cuda)
  set(DIOPI_IMPL_OPT "torch")
elseif(${DEVICE} IN_LIST DEVICE_MUXI)
  set(USE_KUNLUNXIN ON)
  set(UsedVendor muxi)
  set(DIOPI_IMPL_OPT "muxi")
elseif(${DEVICE} IN_LIST DEVICE_CAMB)
  set(USE_CAMB ON)
  set(UsedVendor camb)
  set(DIOPI_IMPL_OPT "camb")
elseif(${DEVICE} IN_LIST DEVICE_ASCEND)
  set(USE_ASCEND ON)
  set(UsedVendor ascend)
  set(DIOPI_IMPL_OPT "ascend")
elseif(${DEVICE} IN_LIST DEVICE_TOPSRIDER)
  set(USE_TOPSRIDER ON)
  set(UsedVendor topsrider)
  set(DIOPI_IMPL_OPT "tops")
elseif(${DEVICE} IN_LIST DEVICE_SUPA)
  set(USE_SUPA ON)
  set(UsedVendor supa)
  set(DIOPI_IMPL_OPT "supa")
  # SUPA DEVICE DOES NOT NEED TO BUILD DIOPI, so set the target to "" to control the workflow.
elseif(${DEVICE} IN_LIST DEVICE_DROPLET)
  set(USE_DROPLET ON)
  set(UsedVendor droplet)
  set(DIOPI_IMPL_OPT "droplet")
elseif(${DEVICE} IN_LIST DEVICE_KUNLUNXIN)
  set(USE_KUNLUNXIN ON)
  set(UsedVendor kunlunxin)
  set(DIOPI_IMPL_OPT "kunlunxin")
else()
  message(
    FATAL_ERROR
      "No implementation module is compiled, cmake requires option -DDEVICE=CAMB or CUDA or ASCEND or SUPA"
  )
endif()

add_compile_definitions(DIPU_VENDOR_NAME=${UsedVendor})
message(STATUS "DIPU_VENDOR_NAME: " ${UsedVendor})

string(TOUPPER ${UsedVendor} UsedVendorUpper)
string(CONCAT DIPU_VENDOR_NAME_FLAG_DEF "DIPU_VENDOR_NAME_" ${UsedVendorUpper})
add_compile_definitions(${DIPU_VENDOR_NAME_FLAG_DEF}=1)
message(STATUS "Vendor name flag definition: " ${DIPU_VENDOR_NAME_FLAG_DEF})

string(REGEX MATCH "[a-z0-9_]+" DIPU_GIT_HASH ${DIPU_GIT_HASH})
execute_process(COMMAND sh -c "git status --porcelain | egrep '^(M| M)' | wc -l"
                OUTPUT_VARIABLE DIPU_MODIFY_LEN)
if(DIPU_MODIFY_LEN GREATER 0)
  set(DIPU_GIT_HASH ${DIPU_GIT_HASH}-dirty)
endif()
message(STATUS "DIPU_GIT_HASH: " ${DIPU_GIT_HASH})
add_compile_definitions(DIPU_GIT_HASH="${DIPU_GIT_HASH}")

# Automatically generate a list of supported diopi functions
execute_process(
  COMMAND
    sh -c
    "grep -Po 'diopi[a-zA-Z0-9]+(?=\\()' ${CMAKE_CURRENT_SOURCE_DIR}/scripts/autogen_diopi_wrapper/diopi_functions.yaml | sort -uf > ${CMAKE_CURRENT_SOURCE_DIR}/SupportedDiopiFunctions.txt"
)

# abi_v, torch dir, abi flag, cmake path
execute_process(
COMMAND
  sh -c
    "python ${CMAKE_CURRENT_SOURCE_DIR}/detect_env.py"
  OUTPUT_VARIABLE CHECK_TORCH_OUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE "\n" ";" CHECK_TORCH_OUT_LIST "${CHECK_TORCH_OUT}")
message(STATUS "CHECK_TORCH_OUT_LIST: " "${CHECK_TORCH_OUT_LIST}")

if(NOT DEFINED DIPU_ABI_V)
  list(GET CHECK_TORCH_OUT_LIST 0 DIPU_ABI_V)
endif()

if(NOT DEFINED PYTORCH_DIR)
  list(GET CHECK_TORCH_OUT_LIST 1 PYTORCH_DIR)
endif()

if(NOT DEFINED DIPU_COMPILED_WITH_CXX11_ABI)
  list(GET CHECK_TORCH_OUT_LIST 2 DIPU_COMPILED_WITH_CXX11_ABI)
endif()

list(GET CHECK_TORCH_OUT_LIST 3 TORCH_CMAKE_PREFIX)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})

list(GET CHECK_TORCH_OUT_LIST 4 DIPU_TORCH_HAS_CUDA)

message(STATUS "DIPU_ABI_V: ${DIPU_ABI_V}")
add_compile_options(-fabi-version=${DIPU_ABI_V})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})

if(DIPU_COMPILED_WITH_CXX11_ABI GREATER 0)
  set(DIPU_COMPILED_WITH_CXX11_ABI 1)
else()
  set(DIPU_COMPILED_WITH_CXX11_ABI 0)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI}")
message(STATUS "DIPU_COMPILED_WITH_CXX11_ABI:" ${DIPU_COMPILED_WITH_CXX11_ABI})

execute_process(COMMAND sh -c "dirname $(find  $(dirname $(which python))/../ -name Python.h)"
                OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
message(STATUS "PYTHON_INCLUDE_DIR: " ${PYTHON_INCLUDE_DIR})

# config
include(cmake/BaseFuncions.cmake)
_set_cpp_flags()

# if add new version, please also update dipu/__init__.py torch_ver_XX
list(APPEND DIPU_SUPPORT_TORCHS "2.0.0" "2.1.0" "2.1.1" "2.2.2")

# use muxi torch. if device is muxi and torch has 'cuda'(muxi).
if (${DEVICE} IN_LIST DEVICE_MUXI AND DIPU_TORCH_HAS_CUDA)
  set(USE_MACA ON)
  # it's used in muxi torch and all targets using torch header file need it.
  add_compile_definitions(USE_MACA=1)
endif()

find_package(Torch REQUIRED)

message(STATUS "Found Torch Version: ${Torch_VERSION}")
if(NOT ${Torch_VERSION} IN_LIST DIPU_SUPPORT_TORCHS)
  message(FATAL_ERROR "this torch version is not supported by DIPU ${DIPU_SUPPORT_TORCHS}")
endif()

# message(STATUS "Torch TORCH_LIBRARY: ${TORCH_LIBRARY}")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch TORCH_CXX_FLAGS:  ${TORCH_CXX_FLAGS}")

string(REGEX REPLACE "^.*(..)\$" "\\1" Torch_VERSION_MINOR "0000${Torch_VERSION_MINOR}")
string(REGEX REPLACE "^.*(..)\$" "\\1" Torch_VERSION_PATCH "0000${Torch_VERSION_PATCH}")

string(CONCAT Torch_VERSION ${Torch_VERSION_MAJOR}${Torch_VERSION_MINOR}${Torch_VERSION_PATCH})
# expand to pattern XYYZZ by leading zero on minor and patch ver
message(STATUS "change to dipu Torch Version: ${Torch_VERSION}")

# add_compile_definitions(DIPU_TORCH${Torch_VERSION})
add_compile_definitions(DIPU_TORCH_VERSION=${Torch_VERSION})

link_directories(${PYTORCH_DIR}/torch/lib)
set(DIPU_TORCH_VERSION ${Torch_VERSION})

# end torch find

if(ENABLE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

add_subdirectory(third_party)

if(LIBS)
  add_subdirectory(torch_dipu/csrc_dipu)
endif()

if(TESTS)
  add_subdirectory(tests/cpp)
endif()
