
cmake_minimum_required(VERSION 3.10)
project(TorchDIPU)

option(TESTS "Whether to build unit tests" OFF)
option(LIBS  "Whether to build dipu lib, default on" ON)

# device related
option(CAMB "Whether to use CambriconNeuware when available" ON)
option(CUDA "Whether to use CUDA when available" OFF)
option(ASCEND "Whether to use ASCEND when available" OFF)
option(STPU "Whether to use STPU when available" OFF)



# config
include(cmake/BaseFuncions.cmake)
_set_cpp_flags()

set(UsedVendor "")
if (CAMB)
  set(USE_CAMB ON)
  set(UsedVendor camb)
endif()

if (CUDA)
  set(USE_CUDA ON)
  set(UsedVendor cuda)
endif()

if (ASCEND)
  set(USE_ASCEND ON)
  set(UsedVendor ascend)
endif()

if (STPU)
  set(USE_STPU ON)
  set(UsedVendor stpu)
endif()



# findTorch encounter some error, need check。
# set(Torch_DIR ${PYTORCH_DIR}/torch/share/cmake/Torch)
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
set(DIOPI_PATH "${PROJECT_SOURCE_DIR}/third_party/DIOPI")
include_directories(${DIOPI_PATH}/include)
add_definitions(-DDIOPI_ATTR_WEAK)

if (LIBS)
  add_subdirectory(torch_dipu/csrc_dipu)
endif()

if (TESTS)
  add_subdirectory(tests/cpp)
endif()






