set(DIPU_LIB torch_dipu)
set(DIPU_PYTHON_LIB torch_dipu_python)

find_package(PythonInterp REQUIRED)
message(STATUS "Python version: " "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# python path
include_directories(${PYTHON_INCLUDE_DIR})

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
    runtime/core/device.cpp
    runtime/core/MemChecker.cpp
    runtime/distributed/*.cpp
    runtime/devproxy/*.cpp
)

file(GLOB OP_SRC_FILES aten/RegisterDIPU.cpp
    aten/CPUFallback.cpp
    aten/util/*.cpp
    aten/ops/*Kernel*.cpp
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
set_target_properties(${DIPU_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/torch_dipu )

# link
target_link_libraries(${DIPU_LIB} ${DIPU_VENDOR_LIB})

if(DEFINED ENV{DIOPI_IMPL_LIB})
  find_library(DIOPI_IMPL_LIB NAMES diopi_impl HINTS ENV DIOPI_IMPL_LIB REQUIRED)
else()
  find_library(DIOPI_IMPL_LIB NAMES diopi_impl HINTS ${DIOPI_IMPL_LIB_PATH} REQUIRED)
endif()

# need export LIBRARY_PATH=$DIOPI_ROOT:$LIBRARY_PATH;
target_link_libraries(${DIPU_LIB}  -Wl,--no-as-needed ${DIOPI_IMPL_LIB} -Wl,--as-needed)
target_link_libraries(${DIPU_LIB}  c10 torch torch_cpu)

## copy vendor header file
set(VENDOR_DIST  "${DIST_DIR}/csrc_dipu/vendor/")
add_custom_target(copy_include
    COMMAND mkdir -p ${VENDOR_DIST}
    COMMAND cmake -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/vendor/${UsedVendor}/vendorapi.h  ${VENDOR_DIST}/vendorapi.h
)

add_custom_command(TARGET copy_include PRE_BUILD    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py --config ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/autogen_diopi_wrapper/diopi_functions.yaml  --out ${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/AutoGenedKernels.cpp  --use_diopi_adapter False  --autocompare False --print_func_call_info True  --print_op_arg True  --fun_config_dict '{\"current_device\": \"${UsedVendor}\"}'
)

add_dependencies(${DIPU_LIB} copy_include)


# --------build bind in python --------------
file(GLOB BIND_SRC_FILES binding/Export*.cpp
)
set(BIND_FILES
    ${BIND_SRC_FILES}
)
add_library(${DIPU_PYTHON_LIB} SHARED ${BIND_SRC_FILES})
set_target_properties(${DIPU_PYTHON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/torch_dipu )
# default hidden setting scope is incorrect and cannot open now because it cause diopirt hidden,
# so temporarily use this target level setting. enhance in future. todo
set_target_properties(${DIPU_PYTHON_LIB} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(${DIPU_PYTHON_LIB} ${DIPU_LIB})


# -------- build _C.python extention --------
execute_process(COMMAND  sh -c "uname -m" OUTPUT_VARIABLE DIPU_C_PYTHON_EXT_ARCH)
string (REPLACE "\n" "" DIPU_C_PYTHON_EXT_ARCH "${DIPU_C_PYTHON_EXT_ARCH}")
string (REPLACE "\r" "" DIPU_C_PYTHON_EXT_ARCH "${DIPU_C_PYTHON_EXT_ARCH}")
set(DIPU_C_PYTHON_EXT_LIB "_C.cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-${DIPU_C_PYTHON_EXT_ARCH}-linux-gnu")
file(GLOB DIPU_C_PYTHON_EXT_SRC_FILES stub.cpp
)
set(DIPU_C_PYTHON_EXT_FILES
    ${DIPU_C_PYTHON_EXT_SRC_FILES}
)
message(STATUS "dipu c python extention building " ${DIPU_C_PYTHON_EXT_FILES})
add_library(${DIPU_C_PYTHON_EXT_LIB} SHARED ${DIPU_C_PYTHON_EXT_FILES})
set_target_properties(${DIPU_C_PYTHON_EXT_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/torch_dipu )
target_link_libraries(${DIPU_C_PYTHON_EXT_LIB} ${DIPU_LIB})
target_link_libraries(${DIPU_C_PYTHON_EXT_LIB} ${DIPU_PYTHON_LIB})
target_link_libraries(${DIPU_C_PYTHON_EXT_LIB} "c10")
target_link_libraries(${DIPU_C_PYTHON_EXT_LIB} "torch")
target_link_libraries(${DIPU_C_PYTHON_EXT_LIB} "torch_cpu")
target_link_libraries(${DIPU_C_PYTHON_EXT_LIB} "torch_python")
target_compile_options(${DIPU_C_PYTHON_EXT_LIB} PRIVATE -fstack-protector-all)
set_target_properties(${DIPU_C_PYTHON_EXT_LIB} PROPERTIES PREFIX "")


# --------- install libs --------------------
message(STATUS "CMake install direcotry:   " ${PROJECT_SOURCE_DIR}/torch_dipu)
