set(DIPU_LIB torch_dipu)
set(DIPU_PYTHON_LIB torch_dipu_python)

# python path
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

# dipu include path
set(DIPU_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${DIPU_PATH})

set(DIST_DIR ${DIPU_PATH}/dist/include)
include_directories(${DIST_DIR})

# src
file(GLOB RT_SRC_FILES
    runtime/core/guardimpl/*.cpp
    runtime/core/allocator/*.cpp
    runtime/core/DIPU*.cpp
    runtime/core/MemChecker.cpp
    runtime/distributed/*.cpp
    runtime/devproxy/*.cpp
)

file(GLOB OP_SRC_FILES aten/RegisterDIPU.cpp
    aten/CPUFallback.cpp
    aten/util/*.cpp
    aten/ops/*Kernel*.cpp
    aten/ops/DIPUAmp.cpp
    aten/ops/CustomFallbackFunctionsForAmpGradScaler.cpp
)

file(GLOB BASE_FILES base/*.cpp)
file(GLOB UTILS_FILES utils/*.cpp)
file(GLOB DIOPI_RT_FILES diopirt/*.cpp)
file(GLOB PROFILER_FILES profiler/*.cpp)

# vendor src
add_subdirectory(vendor/${UsedVendor})
include_directories(SYSTEM ${VENDOR_INCLUDE_DIRS})
link_directories(${VENDOR_LIB_DIRS})


# if (${UsedVendor} STREQUAL ${VendorMLU})
#     message(STATUS "---- vendor specific (but shouldn't exist)----")
# endif()

set(SOURCE_FILES
    ${RT_SRC_FILES}
    ${OP_SRC_FILES}
    ${DIOPI_RT_FILES}
    ${BASE_FILES}
    ${UTILS_FILES}
    ${VENDOR_FILES}
    ${PROFILER_FILES}
)

add_library(${DIPU_LIB} SHARED ${SOURCE_FILES})

# link
target_link_libraries(${DIPU_LIB} ${DIPU_VENDOR_LIB})

target_link_libraries(${DIPU_LIB}  -Wl,--no-as-needed diopi_impl -Wl,--as-needed)
target_link_libraries(${DIPU_LIB}  c10 torch torch_cpu)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${DIPU_LIB} Threads::Threads)

## copy vendor header file
set(VENDOR_DIST "${DIST_DIR}/csrc_dipu/vendor/")
set(VENDOR_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/${UsedVendor}")
add_custom_command(OUTPUT vendor_include
    COMMAND mkdir -p ${VENDOR_DIST}
    COMMAND cmake -E create_symlink ${VENDOR_HEADER_DIR}/vendorapi.h ${VENDOR_DIST}/vendorapi.h)
if (EXISTS "${VENDOR_HEADER_DIR}/vendor_autocast.h" AND
        NOT IS_DIRECTORY "${VENDOR_HEADER_DIR}/vendor_autocast.h")
    add_custom_command(OUTPUT vendor_include APPEND
        COMMAND cmake -E create_symlink ${VENDOR_HEADER_DIR}/vendor_autocast.h ${VENDOR_DIST}/vendor_autocast.h)
else()
    target_compile_definitions(${DIPU_LIB} PRIVATE DIPU_NO_VENDOR_AUTOCAST)
endif()
add_custom_target(copy_include DEPENDS vendor_include)
add_dependencies(${DIPU_LIB} copy_include)


# --------build bind in python --------------
file(GLOB BIND_SRC_FILES binding/Export*.cpp
  binding/patch*.cpp
)
set(BIND_FILES
    ${BIND_SRC_FILES}
)
add_library(${DIPU_PYTHON_LIB} SHARED ${BIND_SRC_FILES})
# default hidden setting scope is incorrect and cannot open now because it cause diopirt hidden,
# so temporarily use this target level setting. enhance in future. todo
set_target_properties(${DIPU_PYTHON_LIB} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(${DIPU_PYTHON_LIB} ${DIPU_LIB})
