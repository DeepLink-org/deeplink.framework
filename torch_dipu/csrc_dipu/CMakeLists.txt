cmake_minimum_required(VERSION 3.10)
project(torch_dipu)
set(DIPU_LIB torch_dipu)

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

# use csrc_dipu as basic include path
set(PROJECT_TOP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../")
include_directories(${PROJECT_TOP_PATH})

list(APPEND TORCH_INCLUDE_DIRS ${PYTORCH_SOURCE_PATH}/torch/include/
    ${PYTORCH_SOURCE_PATH}/torch/include/torch/csrc/api/include/
    )
include_directories(${TORCH_INCLUDE_DIRS})

include_directories(${PYTHON_INCLUDE_DIR})


set(TORCH_LIBRARY_DIR  ${PYTORCH_SOURCE_PATH}/torch/lib)
link_directories(${TORCH_LIBRARY_DIR})


set(DIST_DIR ${CMAKE_SOURCE_DIR}/../dist/include)
include_directories(${DIST_DIR})

# common src
file(GLOB RT_SRC_FILES runtime/core/guardimpl/*.cpp
    runtime/core/allocator.cpp
    runtime/core/DIPU*.cpp
    # runtime/core/DIPUGeneratorImpl.cpp
)

file(GLOB BIND_SRC_FILES binding/Export*.cpp)
file(GLOB OP_SRC_FILES aten/*.cpp)


# vendor src
add_subdirectory(runtime/vendor/${UsedVendor})
include_directories(SYSTEM ${VENDOR_INCLUDE_DIRS})
link_directories(${VENDOR_LIB_DIRS})

# if (${UsedVendor} STREQUAL ${VendorMLU})
#     message(STATUS "---- vendor specific (but shouldn't exist)----")
# endif()

set(SOURCE_FILES
    ${RT_SRC_FILES}
    ${BIND_SRC_FILES}
    ${OP_SRC_FILES}
    ${VENDOR_FILES}

)

add_library(${DIPU_LIB} SHARED ${SOURCE_FILES})

# link
target_link_libraries(${DIPU_LIB} cnnl cncl)
target_link_libraries(${DIPU_LIB} c10 torch torch_cpu)

## copy vendor header file
set(VENDOR_DIST  "${DIST_DIR}/csrc_dipu/runtime/vendor/")
add_custom_target(copy_include
    COMMAND mkdir -p ${VENDOR_DIST}
    COMMAND cp ${CMAKE_SOURCE_DIR}/runtime/vendor/${UsedVendor}/vendorapi.h  ${VENDOR_DIST}
)
add_dependencies(${DIPU_LIB} copy_include)





