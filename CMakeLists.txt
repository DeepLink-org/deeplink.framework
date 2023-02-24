cmake_minimum_required(VERSION 3.10)
project(torch_dipu)
set(DIPU_LIB torch_dipu)
set(DIPU_PYTHON_LIB torch_dipu_python)

set(VendorMLU mlu)
set(VendorAscend ascend)
set(UsedVendor ${VendorMLU})

# config
set(CMAKE_BUILD_TYPE Debug)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fPIC")
# SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}  -g -O0 -fPIC")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wreturn-type -Wno-error=deprecated-declarations ")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
# symbol hidden
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(GLIBCXX_USE_CXX11_ABI 1)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

if(NOT DEFINED PYTORCH_SOURCE_PATH)
    message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

# should use findTorch
set(TORCH_LIBRARY_DIR "${PYTORCH_SOURCE_PATH}/torch/lib")
set(TORCH_LIBRARY_DIR  ${PYTORCH_SOURCE_PATH}/torch/lib)
link_directories(${TORCH_LIBRARY_DIR})

list(APPEND TORCH_INCLUDE_DIRS ${PYTORCH_SOURCE_PATH}/torch/include/
    ${PYTORCH_SOURCE_PATH}/torch/include/torch/csrc/api/include/
    )
include_directories(${TORCH_INCLUDE_DIRS})

add_subdirectory(torch_dipu/csrc_dipu)

add_subdirectory(tests/cpp)





