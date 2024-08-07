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

message(STATUS "VENDOR_INCLUDE_DIRS: ${VENDOR_INCLUDE_DIRS}")
message(STATUS "VENDOR_LIB_DIRS:  ${VENDOR_LIB_DIRS}")
message(STATUS "DIPU_VENDOR_LIB:  ${DIPU_VENDOR_LIB}")

# TODO(vendor) - Refactor those command after vendor is ready.
set(VENDOR_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dist/include")
set(VENDOR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/${UsedVendor}")
set(VENDOR_OUTPUT_HEADER "${VENDOR_DIST_DIR}/csrc_dipu/vendor/vendorapi.h")
add_custom_command(
  OUTPUT "${VENDOR_OUTPUT_HEADER}"
  COMMAND mkdir -p "${VENDOR_DIST_DIR}/csrc_dipu/vendor"
  COMMAND cmake -E create_symlink
    "${VENDOR_SOURCE_DIR}/vendorapi.h"
    "${VENDOR_DIST_DIR}/csrc_dipu/vendor/vendorapi.h"
  VERBATIM)

if(EXISTS "${VENDOR_SOURCE_DIR}/vendor_autocast.h")
  add_custom_command(
    OUTPUT "${VENDOR_OUTPUT_HEADER}"
    COMMAND cmake -E create_symlink
      "${VENDOR_SOURCE_DIR}/vendor_autocast.h"
      "${VENDOR_DIST_DIR}/csrc_dipu/vendor/vendor_autocast.h"
    APPEND
    VERBATIM)
endif()

#[[ Target: torch_dipu ]]

# Auto generated code.
set(AUTOGEN_DIOPI_WRAPPER_DIR "${PROJECT_SOURCE_DIR}/scripts/autogen_diopi_wrapper")
set(AUTOGEN_CODE_SH "${AUTOGEN_DIOPI_WRAPPER_DIR}/autogen_wrapped_code.sh")

# The following parameters are used to auto_generated code.
set(GENERATED_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/AutoGenedKernels.cpp")
set(GENERATED_KERNELS_VENDOR "${PROJECT_SOURCE_DIR}/third_party/DIOPI/impl/${UsedVendor}/convert_config.yaml")
set(GENERATED_KERNELS_SCRIPT "${AUTOGEN_DIOPI_WRAPPER_DIR}/autogen_diopi_wrapper.py")
set(GENERATED_KERNELS_CONFIG "${AUTOGEN_DIOPI_WRAPPER_DIR}/diopi_functions.yaml")
set(DEVICE_GUARD_FREE_VENDOR "") # Vendors that do not require Device Guard should be included in this list.
if (${UsedVendor} IN_LIST DEVICE_GUARD_FREE_VENDOR)
  set(GENERATE_DEVICE_GUARD False)
else()
  set(GENERATE_DEVICE_GUARD True)
endif()
message("GENERATE_DEVICE_GUARD: " ${GENERATE_DEVICE_GUARD})

if(NOT EXISTS "${GENERATED_KERNELS_VENDOR}")
  unset(GENERATED_KERNELS_VENDOR)
endif()

add_custom_command(
  OUTPUT "${GENERATED_KERNELS}"
  COMMAND bash -c "${AUTOGEN_CODE_SH} ${UsedVendor} ${Torch_VERSION} ${GENERATED_KERNELS_SCRIPT} ${GENERATED_KERNELS_CONFIG} ${GENERATED_KERNELS} ${GENERATE_DEVICE_GUARD} ${ENABLE_DIPU_EXTRA_FEATURE}"
  COMMENT "Generating ${GENERATED_KERNELS}$<$<BOOL:${GENERATED_KERNELS_VENDOR}>: with ${GENERATED_KERNELS_VENDOR}>"
  DEPENDS
    "${GENERATED_KERNELS_SCRIPT}"
    "${GENERATED_KERNELS_CONFIG}"
    "${AUTOGEN_CODE_SH}"
  )

# Collect source files.
# You may use: find . -name "*.cpp" to update list.
set(TORCH_DIPU_SOURCE
  aten/ops/CustomFallbackFunctionsForAmpGradScaler.cpp
  aten/ops/DIPUCopy.cpp
  aten/ops/StorageShapeKernel.cpp
  aten/ops/DIPUAmp.cpp
  aten/ops/DIPUOpInferrer.cpp
  aten/ops/PinMemoryKernel.cpp
  aten/ops/EmptyOpsKernel.cpp
  aten/ops/CustomFallbackFunctionsForCopy.cpp
  aten/ops/OpRegexMatch.cpp
  aten/RegisterDIPU.cpp
  aten/CPUFallback.cpp

  base/DIPUGlobals.cpp

  diopirt/diopirt_impl.cpp
  diopirt/diopi_helper.cpp

  metrics/metrics.cpp

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
  runtime/core/allocator/DIPUCachingHostAllocator.cpp
  runtime/core/allocator/DIPUCachingDeviceAllocator.cpp
  runtime/core/MemChecker.cpp
  runtime/core/guardimpl/DIPUGuardImpl.cpp
  runtime/core/DIPUGeneratorImpl.cpp
  runtime/core/DIPUStream.cpp
  runtime/device/deviceapis.cpp

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

target_link_libraries(torch_dipu PRIVATE Python3::Python torch)
target_link_libraries(torch_dipu PRIVATE Threads::Threads)
target_link_libraries(torch_dipu PRIVATE diopi)

#[[ Target: torch_dipu_python ]]
set(TORCH_DIPU_PYBIND_SOURCE
  binding/ExportRT.cpp
  binding/ExportProfiler.cpp
  binding/patchCsrcDevice.cpp
  binding/ExportTensor.cpp)

Python3_add_library(torch_dipu_python SHARED ${TORCH_DIPU_PYBIND_SOURCE})

# TODO: default hidden setting scope is incorrect and cannot open now because it
# cause diopirt hidden, so temporarily use this target level setting. enhance in
# future.
set_target_properties(torch_dipu_python PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torch_dipu_python PUBLIC torch_dipu PRIVATE kineto)
target_include_directories(torch_dipu_python SYSTEM PRIVATE ..)

#[[ Target: torch_dipu_cpython_extension ]]
Python3_add_library(torch_dipu_cpython_extension SHARED stub.cpp)
target_include_directories(torch_dipu_cpython_extension SYSTEM PRIVATE ..)
target_link_libraries(torch_dipu_cpython_extension PUBLIC torch_dipu_python PRIVATE torch_python)
target_compile_options(torch_dipu_cpython_extension PRIVATE -fstack-protector-all)
set_target_properties(torch_dipu_cpython_extension PROPERTIES
  OUTPUT_NAME "_C.cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
  PREFIX "")

set_target_properties(torch_dipu torch_dipu_python torch_dipu_cpython_extension
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/torch_dipu")

# TODO(lljbash,lihuayi): set the lib output dir like pytorch
