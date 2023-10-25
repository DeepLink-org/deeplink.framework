
cmake_minimum_required(VERSION 3.10)
project(TorchDIPU)

option(TESTS "Whether to build unit tests" OFF)
option(LIBS  "Whether to build dipu lib, default on" ON)

# use gcover
OPTION (ENABLE_COVERAGE "Use gcov" OFF)
MESSAGE(STATUS ENABLE_COVERAGE=${ENABLE_COVERAGE})

# device related
set(DEVICE "camb" CACHE STRING "device string, default camb")
list(APPEND DEVICE_CAMB "CAMB" "camb")
list(APPEND DEVICE_CUDA "CUDA" "cuda")
list(APPEND DEVICE_ASCEND "ASCEND" "ascend")
list(APPEND DEVICE_TOPSRIDER "TOPS" "tops" "TOPSRIDER" "topsrider")
list(APPEND DEVICE_SUPA "SUPA" "supa")
list(APPEND DEVICE_DROPLET "DROPLET" "droplet")

execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE DIPU_GIT_HASH)
string(REGEX MATCH "[a-z0-9_]+" DIPU_GIT_HASH ${DIPU_GIT_HASH})
execute_process(COMMAND  sh -c "git status --porcelain | egrep '^(M| M)' | wc -l" OUTPUT_VARIABLE DIPU_MODIFY_LEN)
if (DIPU_MODIFY_LEN GREATER 0)
    set(DIPU_GIT_HASH ${DIPU_GIT_HASH}-dirty)
endif()
message(STATUS "DIPU_GIT_HASH: " ${DIPU_GIT_HASH})
add_compile_options(-DDIPU_GIT_HASH="${DIPU_GIT_HASH}")

# config
include(cmake/BaseFuncions.cmake)
_set_cpp_flags()

set(UsedVendor "")
if (${DEVICE} IN_LIST DEVICE_CUDA)
    set(USE_CUDA ON)
    set(UsedVendor cuda)
elseif (${DEVICE} IN_LIST DEVICE_CAMB)
    set(USE_CAMB ON)
    set(UsedVendor camb)
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



if (ENABLE_COVERAGE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()


# findTorch encounter some error, need check。
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


set(TORCH_LIBRARY_DIR "${PYTORCH_DIR}/torch/lib")
link_directories(${TORCH_LIBRARY_DIR})
list(APPEND TORCH_INCLUDE_DIRS ${PYTORCH_DIR}/torch/include/
    ${PYTORCH_DIR}/torch/include/torch/csrc/api/include/
)
include_directories(${TORCH_INCLUDE_DIRS})
message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

# diopi
set(DIOPI_PROTO_PATH "${PROJECT_SOURCE_DIR}/third_party/DIOPI/proto")
include_directories(${DIOPI_PROTO_PATH}/include)
add_definitions(-DDIOPI_ATTR_WEAK)

if (LIBS)
  add_subdirectory(torch_dipu/csrc_dipu)
endif()

if (TESTS)
  add_subdirectory(tests/cpp)
endif()





