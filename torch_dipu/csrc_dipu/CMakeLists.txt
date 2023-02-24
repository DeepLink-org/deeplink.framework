# python path
include_directories(${PYTHON_INCLUDE_DIR})

# use csrc_dipu as basic include path
set(TORCH_DIPU_TOP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../")
include_directories(${TORCH_DIPU_TOP_PATH})


set(DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dist/include)
include_directories(${DIST_DIR})

# common src
file(GLOB RT_SRC_FILES runtime/core/guardimpl/*.cpp
    runtime/core/allocator.cpp
    runtime/core/DIPU*.cpp
    # runtime/core/DIPUGeneratorImpl.cpp
)
file(GLOB OP_SRC_FILES aten/RegisterDIPU.cpp
        aten/ops/*Kernel.cpp
    )

# vendor src
add_subdirectory(runtime/vendor/${UsedVendor})
include_directories(SYSTEM ${VENDOR_INCLUDE_DIRS})
link_directories(${VENDOR_LIB_DIRS})

# if (${UsedVendor} STREQUAL ${VendorMLU})
#     message(STATUS "---- vendor specific (but shouldn't exist)----")
# endif()

set(SOURCE_FILES
    common.cpp
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
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/runtime/vendor/${UsedVendor}/vendorapi.h  ${VENDOR_DIST}
)
add_dependencies(${DIPU_LIB} copy_include)

# build bind
file(GLOB BIND_SRC_FILES binding/Export*.cpp)
set(BIND_FILES
    ${BIND_SRC_FILES}
)
add_library(${DIPU_PYTHON_LIB} SHARED ${BIND_SRC_FILES})
target_link_libraries(${DIPU_PYTHON_LIB} ${DIPU_LIB})



