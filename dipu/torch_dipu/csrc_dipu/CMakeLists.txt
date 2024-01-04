#[[ Dependencies ]]

# Import Python3::Python, Python3_EXECUTABLE
# Also see https://cmake.org/cmake/help/latest/module/FindPython3.html
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# See https://pybind11.readthedocs.io/en/stable/cmake/index.html
# Note: It seems strange that we are not using pybind11.
# find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# See https://cmake.org/cmake/help/latest/module/FindThreads.html
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Vendor
# Import VENDOR_INCLUDE_DIRS, VENDOR_LIB_DIRS, VENDOR_FILES, DIPU_VENDOR_LIB (optional)
# TODO(vendor) - make vendor targets.
add_subdirectory(vendor/${UsedVendor})

# TODO(vendor) - Refactor those command after vendor is ready.
set(VENDOR_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dist/include")
set(VENDOR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/${UsedVendor}")
set(VENDOR_OUTPUT_HEADER "${VENDOR_DIST_DIR}/csrc_dipu/vendor/vendorapi.h")
add_custom_command(
  OUTPUT ${VENDOR_OUTPUT_HEADER}
  COMMAND mkdir -p ${VENDOR_DIST_DIR}/csrc_dipu/vendor
  COMMAND cmake -E create_symlink
  ${VENDOR_SOURCE_DIR}/vendorapi.h
  ${VENDOR_DIST_DIR}/csrc_dipu/vendor/vendorapi.h
  VERBATIM)

if(EXISTS "${VENDOR_SOURCE_DIR}/vendor_autocast.h")
  add_custom_command(
    OUTPUT ${VENDOR_OUTPUT_HEADER}
    COMMAND cmake -E create_symlink
    ${VENDOR_SOURCE_DIR}/vendor_autocast.h
    ${VENDOR_DIST_DIR}/csrc_dipu/vendor/vendor_autocast.h
    APPEND
    VERBATIM)
endif()

#[[ Target: torch_dipu ]]

# Auto generated code.
# Consider moving it to other files.
set(GENERATED_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/AutoGenedKernels.cpp")
set(GENERATED_KERNELS_SCRIPT "${PROJECT_SOURCE_DIR}/scripts/autogen_diopi_wrapper/autogen_diopi_wrapper.py")
set(GENERATED_KERNELS_CONFIG "${PROJECT_SOURCE_DIR}/scripts/autogen_diopi_wrapper/diopi_functions.yaml")
set(GENERATED_KERNELS_VENDOR "${PROJECT_SOURCE_DIR}/third_party/DIOPI/impl/${UsedVendor}/convert_config.yaml")

if(NOT EXISTS "${GENERATED_KERNELS_VENDOR}")
  unset(GENERATED_KERNELS_VENDOR)
endif()

add_custom_command(
  OUTPUT "${GENERATED_KERNELS}"
  COMMAND "${Python3_EXECUTABLE}"
  "${GENERATED_KERNELS_SCRIPT}"
  --out=${GENERATED_KERNELS}
  --config=${GENERATED_KERNELS_CONFIG}
  --autocompare=False
  --print_op_arg=True
  --use_diopi_adapter=False
  --print_func_call_info=True
  --fun_config_dict={\"current_device\":\"${UsedVendor}\"}
  $<$<BOOL:GENERATED_KERNELS_VENDOR>:--convert_config=${GENERATED_KERNELS_VENDOR}>
  DEPENDS "${GENERATED_KERNELS_SCRIPT}"
  "${GENERATED_KERNELS_CONFIG}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generate file ${GENERATED_KERNELS} in ${CMAKE_CURRENT_SOURCE_DIR}"
  VERBATIM)

# Collect source files.
# Tou may use: find . -name "*.cpp" to update.
set(TORCH_DIPU_SOURCE
  aten/ops/CustomFallbackFunctionsForAmpGradScaler.cpp
  aten/ops/DIPUCopy.cpp
  aten/ops/StorageShapeKernel.cpp
  aten/ops/DIPUAmp.cpp
  aten/ops/PinMemoryKernel.cpp
  aten/ops/EmptyOpsKernel.cpp
  aten/ops/CustomFallbackFunctionsForCopy.cpp
  aten/RegisterDIPU.cpp
  aten/CPUFallback.cpp

  base/DIPUGlobals.cpp

  diopirt/diopirt_impl.cpp
  diopirt/diopi_helper.cpp

  profiler/collection.cpp
  profiler/CorrelationIDManager.cpp
  profiler/profiler.cpp
  profiler/profiler_python.cpp
  profiler/profiler_kineto.cpp
  profiler/DIPUDeviceActivity.cpp
  profiler/patch.cpp

  runtime/distributed/ProcessGroupDICL.cpp
  runtime/distributed/c10dOps.cpp
  runtime/devproxy/deviceproxy.cpp
  runtime/devproxy/diclproxy.cpp
  runtime/core/DIPUEventPool.cpp
  runtime/core/DIPUDeviceInfo.cpp
  runtime/core/allocator/DIPURawCachingAllocator.cpp
  runtime/core/allocator/DIPURawAllocator.cpp
  runtime/core/allocator/DIPUCachingAllocator.cpp
  runtime/core/allocator/DIPUBFCachingAllocator.cpp
  runtime/core/allocator/DIPUBSCachingAllocator.cpp
  runtime/core/MemChecker.cpp
  runtime/core/guardimpl/DIPUGuardImpl.cpp
  runtime/core/DIPUGeneratorImpl.cpp
  runtime/core/DIPUStream.cpp

  utils/helpfunc.cpp)

add_library(torch_dipu SHARED ${TORCH_DIPU_SOURCE} ${GENERATED_KERNELS} ${VENDOR_FILES} ${VENDOR_OUTPUT_HEADER})
target_include_directories(torch_dipu PUBLIC ..)

# TODO(vendor) - replace those code with target_link_libraries.
target_include_directories(torch_dipu SYSTEM PUBLIC ${VENDOR_INCLUDE_DIRS} ${VENDOR_DIST_DIR})
target_link_directories(torch_dipu PRIVATE ${VENDOR_LIB_DIRS})
target_link_libraries(torch_dipu PRIVATE ${DIPU_VENDOR_LIB})

if(NOT EXISTS "${VENDOR_SOURCE_DIR}/vendor_autocast.h")
  target_compile_definitions(torch_dipu PRIVATE DIPU_NO_VENDOR_AUTOCAST)
endif()

# Note for kineto:
# Target kineto only contains object files. Thus we need to do something to
# fetch header files. And kineto's public headers are also used by other
# target, so they are marked as PUBLIC.
target_link_libraries(torch_dipu PRIVATE kineto)
target_compile_definitions(torch_dipu PRIVATE USE_KINETO)
target_include_directories(torch_dipu SYSTEM
  PUBLIC "${kineto_SOURCE_DIR}/include"
  PRIVATE "${kineto_SOURCE_DIR}/src")

target_link_libraries(torch_dipu PRIVATE diopi_impl)
target_link_libraries(torch_dipu PRIVATE Python3::Python c10 torch torch_cpu)
target_link_libraries(torch_dipu PRIVATE Threads::Threads)

#[[ Target: torch_dipu_pybind ]]
set(TORCH_DIPU_PYBIND_SOURCE
  binding/ExportRT.cpp
  binding/ExportProfiler.cpp
  binding/patchCsrcDevice.cpp
  binding/ExportTensor.cpp)

add_library(torch_dipu_pybind ${TORCH_DIPU_PYBIND_SOURCE})

# TODO: default hidden setting scope is incorrect and cannot open now because it
# cause diopirt hidden, so temporarily use this target level setting. enhance in
# future.
set_target_properties(torch_dipu_pybind PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torch_dipu_pybind Python3::Python torch kineto torch_dipu)
target_include_directories(torch_dipu_pybind SYSTEM PUBLIC ..)

#[[ Target: torch_dipu_cpython_extension ]]
Python3_add_library(torch_dipu_cpython_extension SHARED stub.cpp)
target_include_directories(torch_dipu_cpython_extension SYSTEM PUBLIC ..)
target_link_libraries(torch_dipu_cpython_extension PRIVATE torch torch_dipu_pybind torch_dipu)
target_compile_options(torch_dipu_cpython_extension PRIVATE -fstack-protector-all)
set_target_properties(torch_dipu_cpython_extension PROPERTIES
  OUTPUT_NAME "_C.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
  PREFIX "")

set_target_properties(torch_dipu torch_dipu_pybind torch_dipu_cpython_extension
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/torch_dipu")

# TODO(lljbash,lihuayi): set the lib output dir like pytorch
