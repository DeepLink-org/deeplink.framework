
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


execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE DIPU_GIT_HASH)
string(REGEX MATCH "[a-z0-9_]+" DIPU_GIT_HASH ${DIPU_GIT_HASH})
execute_process(COMMAND  sh -c "git status --porcelain | egrep '^(M| M)' | wc -l" OUTPUT_VARIABLE DIPU_MODIFY_LEN)
if (DIPU_MODIFY_LEN GREATER 0)
    set(DIPU_GIT_HASH ${DIPU_GIT_HASH}-dirty)
endif()
message(STATUS "DIPU_GIT_HASH: " ${DIPU_GIT_HASH})
add_compile_options(-DDIPU_GIT_HASH="${DIPU_GIT_HASH}")

# Automatically generate a list of supported diopi functions
execute_process(COMMAND  sh -x -c "grep -Po 'diopi[a-zA-Z0-9]+(?=\\()' ${CMAKE_CURRENT_SOURCE_DIR}/scripts/autogen_diopi_wrapper/diopi_functions.yaml | sort -u > ${CMAKE_CURRENT_SOURCE_DIR}/SupportedDiopiFunctions.txt")

if(SKBUILD)
    #set ABI VERSION with a default value 11, should find a way to fix this later.
    set(DIPU_ABI_V 11)
else()
    execute_process(COMMAND sh -x -c "python -c 'import torch, builtins; print(next(item[-4:-2] for item in dir(builtins)      \
        if \"__pybind11_internals_v4_gcc_libstdcpp_cxxabi10\" in item))'" OUTPUT_VARIABLE DIPU_ABI_V)
endif()
add_definitions(-fabi-version=${DIPU_ABI_V})
message(STATUS "DIPU_ABI_V: ${DIPU_ABI_V}")

if (SKBUILD)
    if (DEFINED ENV{DIPU_COMPILED_WITH_CXX11_ABI})
    #under some enviroment, it failed to detect the CXX11_ABI, while the libtorch_python.so use CXX11_ABI, so we use ENV variable instead.
        if ($ENV{DIPU_COMPILED_WITH_CXX11_ABI} GREATER 0)
            set(DIPU_COMPILED_WITH_CXX11_ABI 1)
        else()
            set(DIPU_COMPILED_WITH_CXX11_ABI 0)
        endif()
    else()
    #As the normal libtorch.so is built with CXX11 ABI, we set the default value as 1
        set(DIPU_COMPILED_WITH_CXX11_ABI 1)
    endif()
else()
    execute_process(COMMAND  sh -x -c "python -c 'import torch;print(1 if torch.compiled_with_cxx11_abi() else 0)'" OUTPUT_VARIABLE DIPU_COMPILED_WITH_CXX11_ABI)
    if (DIPU_COMPILED_WITH_CXX11_ABI GREATER 0)
        set(DIPU_COMPILED_WITH_CXX11_ABI 1)
    else()
        set(DIPU_COMPILED_WITH_CXX11_ABI 0)
    endif()
endif()

<<<<<<< HEAD
=======
if (SKBUILD)
    if ($ENV{DIPU_COMPILED_WITH_CXX11_ABI})
    #under some enviroment, it failed to detect the CXX11_ABI, while the libtorch_python.so use CXX11_ABI, so we use ENV variable instead.
        if ($ENV{DIPU_COMPILED_WITH_CXX11_ABI} GREATER 0)
            set(DIPU_COMPILED_WITH_CXX11_ABI 1)
        else()
            set(DIPU_COMPILED_WITH_CXX11_ABI 0)
        endif()
    else()
    #As the normal libtorch.so is built with CXX11 ABI, we set the default value as 1
        set(DIPU_COMPILED_WITH_CXX11_ABI 1)
    endif()
endif()

>>>>>>> ea84a291f7f7fde0750a4432a53e415e43fc2026
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI}")
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI})
message(STATUS "DIPU_COMPILED_WITH_CXX11_ABI:" ${DIPU_COMPILED_WITH_CXX11_ABI})

execute_process(COMMAND  sh -c "which python" OUTPUT_VARIABLE PYTHON_PATH)
execute_process(COMMAND  sh -c "dirname $(find  $(dirname $(which python))/../ -name Python.h)" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)


message(STATUS "PATH:  " $ENV{PATH})
message(STATUS "PYTHON_INCLUDE_DIR: " ${PYTHON_INCLUDE_DIR})

execute_process(COMMAND  sh -c "dirname $(python -c 'import torch;print(torch.__path__[0])')" OUTPUT_VARIABLE PYTORCH_DIR)
string (REPLACE "\n" "" PYTORCH_DIR "${PYTORCH_DIR}")
string (REPLACE "\r" "" PYTORCH_DIR "${PYTORCH_DIR}")
if(SKBUILD)
    if(DEFINED ENV{PYTORCH_DIR})
        set(PYTORCH_DIR $ENV{PYTORCH_DIR})
    elseif(NOT PYTORCH_DIR)
        message(FATAL_ERROR "YOU MUST SET ENVIROMENT VARIABLE \"PYTORCH_DIR\" to pytorch directory")
    endif()
endif()
message(STATUS "PYTORCH_DIR: " ${PYTORCH_DIR})

# config
include(cmake/BaseFuncions.cmake)
_set_cpp_flags()


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
set(DIOPI_IMPL_LIB_PATH "${PROJECT_SOURCE_DIR}/third_party/DIOPI/impl/lib")
set(DIOPI_IMPL_PATH "${PROJECT_SOURCE_DIR}/third_party/DIOPI/impl/")
include_directories(${DIOPI_PROTO_PATH}/include)
add_definitions(-DDIOPI_ATTR_WEAK)

add_custom_target(DIOPI_LIB ALL)
#--------------- build_DIOPI----------
if(${BUILD_DIOPI})
    #add_custom_command(TARGET DIOPI_LIB PRE_BUILD
    #    COMMAND sh -c "cd ${DIOPI_IMPL_PATH} && sh scripts/build_impl.sh clean"
    #    COMMAND sh -c "cd ${DIOPI_IMPL_PATH} && sh scripts/build_impl.sh ${Build_DIOPI_Target} || exit -1;"
    #    COMMAND sh -c "cd ${CMAKE_CURRENT_SOURCE_DIR};"
    #)
    #There is something wrong "Make tools can not find the build_impl.sh" when use the code above
    execute_process(COMMAND sh -c "cd ${DIOPI_IMPL_PATH}; sh scripts/build_impl.sh clean; sh scripts/build_impl.sh ${Build_DIOPI_Target} || exit -1; cd ../../../;")
    message(STATUS "DIOPI LIB BUILT")
endif()
#---------------------------------------

if (LIBS)
  add_subdirectory(torch_dipu/csrc_dipu)
endif()

if (TESTS)
  add_subdirectory(tests/cpp)
endif()






