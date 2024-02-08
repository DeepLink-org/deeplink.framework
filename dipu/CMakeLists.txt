cmake_minimum_required(VERSION 3.22)
project(torch_dipu LANGUAGES CXX)

#[[ Options ]]
# Options could be listed in CMake GUI and CMakeCache.
option(DIPU_ENABLE_TESTS "Build unit tests" OFF)
option(DIPU_ENABLE_COVERAGE "Build with gcov" OFF)
set(DEVICE "camb" CACHE STRING "Select a device provider") # Consider renaming

# Setup global variable.
set(CMAKE_CXX_EXTENSIONS OFF)

#[[ Helper functions and public dependence ]]
include(cmake/BaseFuncions.cmake)

# Import Python3::Python, Python3_EXECUTABLE
# References: https://cmake.org/cmake/help/latest/module/FindPython3.html
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Import Torch
# - Try also searching PyTorch from Python3_SITELIB and PYTHONPATH.
string(REPLACE ":" ";" DIPU_TORCH_SEARCH_PATH "${Python3_SITELIB}:$ENV{PYTHONPATH}")
find_package(Torch REQUIRED PATHS ${DIPU_TORCH_SEARCH_PATH})

# - Assert Torch version is supported.
# - if adding new versions, please also update dipu/__init__.py torch_ver_XX
set(SUPPORTED_TORCH_VERSIONS 2.0.0 2.1.1)

if(NOT Torch_VERSION IN_LIST SUPPORTED_TORCH_VERSIONS)
  message(FATAL_ERROR "Torch ${Torch_VERSION} is currently unsupported.")
endif()

unset(SUPPORTED_TORCH_VERSIONS)

#[[ Generate files ]]
# Automatically generate a list of supported diopi functions.
generate_supported_diopi_functions(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/autogen_diopi_wrapper/diopi_functions.yaml"
  "${CMAKE_CURRENT_SOURCE_DIR}/SupportedDiopiFunctions.txt")

# config
# ?
_set_cpp_flags()

if(DIPU_ENABLE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# Detect device.
detect_device_name(${DEVICE} DEVICE_PROVIDER)

#[[ Subdirectories ]]
add_subdirectory(third_party)
add_subdirectory(torch_dipu/csrc_dipu)

if(DIPU_ENABLE_TESTS)
  add_subdirectory(tests/cpp)
endif()
