
cmake_minimum_required(VERSION 3.10)
project(TorchDIPU)

option(TESTS "Whether to build unit tests" OFF)
# device related
option(CAMB "Whether to use CambriconNeuware when available" ON)
option(CUDA "Whether to use CUDA when available" OFF)
option(ASCEND "Whether to use ASCEND when available" OFF)


  # config
set(CMAKE_BUILD_TYPE Debug)

include(cmake/BaseFuncions.cmake)
_set_cpp_flags()

set(UsedVendor "")
if (CAMB)
  set(USE_CAMB ON)
  set(UsedVendor camb)
endif()

if (CUDA)
  set(USE_CUDA ON)
  set(UsedVendor CUDA)
endif()

if (ASCEND)
  set(USE_ASCEND ON)
  set(UsedVendor ascend)
endif()

set(Torch_DIR ${PYTORCH_DIR}/torch/share/cmake/Torch)
link_directories(${PYTORCH_DIR}/torch/lib)
find_package(Torch REQUIRED)

if (NOT Torch_FOUND)
    message(FATAL_ERROR "libtorch is required but not found")
else()    
    include_directories(${TORCH_INCLUDE_DIRS})
    message(STATUS "Found Torch Version: ${Torch_VERSION}")
    message(STATUS "Torch TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
endif()


# diopi
set(DIOPI_PATH "${PROJECT_SOURCE_DIR}/third_party/DIOPI")
include_directories(${DIOPI_PATH}/include)
add_definitions(-DDIOPI_ATTR_WEAK)

add_subdirectory(torch_dipu/csrc_dipu)

if (TESTS)
  add_subdirectory(tests/cpp)
endif()






