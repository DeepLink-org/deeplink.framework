#require the external project to build as target
include(ExternalProject)

add_library(diopi_impl SHARED IMPORTED GLOBAL)

if((${WITH_DIOPI} STREQUAL "INTERNAL") OR (${WITH_DIOPI} STREQUAL "internal"))
  if(NOT ${DIOPI_IMPL_OPT} STREQUAL "")
#----------------------------------Build DIOPI submodule-------------------------------------------
    if(NOT DEFINED DIOPI_CMAKE_PREFIX_PATH)
      execute_process(
        COMMAND
          sh -x -c
          "python -c 'import torch;print(torch.utils.cmake_prefix_path)'"
        OUTPUT_VARIABLE DIOPI_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
  
    message(STATUS "building internal DIOPI")
    set(DIOPI_DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third_party")
    set(DIOPI_SRC_PATH "${PROJECT_SOURCE_DIR}/third_party/DIOPI")
    set(DIOPI_BUILD_PATH "${DIOPI_SRC_PATH}/build")
    set(DIOPI_PROTO_PATH "${DIOPI_SRC_PATH}/proto")
    set(DIOPI_IMPL_LIB_PATH "${DIOPI_SRC_PATH}/impl/lib")
    set(DIOPI_IMPL_LIB "${DIOPI_IMPL_LIB_PATH}/libdiopi_impl.so")
    ExternalProject_Add(diopi_internal
                        DOWNLOAD_DIR ${DIOPI_DOWNLOAD_PATH}
                        SOURCE_DIR ${DIOPI_SRC_PATH}
                        SOURCE_SUBDIR impl
                        BINARY_DIR ${DIOPI_BUILD_PATH}
                        DOWNLOAD_COMMAND ""
                        CMAKE_ARGS -DIMPL_OPT=${DIOPI_IMPL_OPT}
                                   -DENABLE_COVERAGE=${USE_COVERAGE} 
                                   -DCMAKE_PREFIX_PATH=${DIOPI_CMAKE_PREFIX_PATH}
                        BUILD_BYPRODUCTS ${DIOPI_IMPL_LIB}
                        INSTALL_COMMAND cmake -E echo "Skipping install step for diopi_internal."
    )
    ## The following code is a work around to avoid make file to run multiple externalProject-build when using make -j N
    ExternalProject_Add_StepTargets(diopi_internal configure build install)
    ExternalProject_Add_StepDependencies(diopi_internal install diopi_internal-build)
    ExternalProject_Add_StepDependencies(diopi_internal build diopi_internal-configure)
    add_dependencies(diopi_impl diopi_internal-install)
  endif()
#--------------------------------------------------------------------------------------------------
else()
  if(EXISTS $ENV{DIOPI_PATH})
    set(DIOPI_PROTO_PATH $ENV{DIOPI_PATH})
  else()
    set(DIOPI_PROTO_PATH "${WITH_DIOPI}/proto")
  endif()
  if(EXISTS $ENV{DIOPI_ROOT})
    set(DIOPI_IMPL_LIB_PATH $ENV{DIOPI_ROOT})
    set(DIOPI_IMPL_LIB "${DIOPI_IMPL_LIB_PATH}/libdiopi_impl.so")
  else()
    set(DIOPI_IMPL_LIB_PATH "${WITH_DIOPI}/impl/lib")
    set(DIOPI_IMPL_LIB "${DIOPI_IMPL_LIB_PATH}/libdiopi_impl.so")
  endif()
endif()

message(STATUS "DIOPI_PROTO_PATH: ${DIOPI_PROTO_PATH}")
message(STATUS "DIOPI_IMPL_LIB_PATH: ${DIOPI_IMPL_LIB_PATH}")

set_target_properties(
  diopi_impl PROPERTIES IMPORTED_LOCATION
                        ${DIOPI_IMPL_LIB})
target_include_directories(diopi_impl SYSTEM INTERFACE ${DIOPI_PROTO_PATH}/include)
target_compile_definitions(diopi_impl INTERFACE DIOPI_ATTR_WEAK)
#-----------------------------------------------------------------------------------------------

#-------------------------add kineto as an external project ------------------------------------
#-------------------------use the local submodule(without download)-----------------------------
set(KINETO_BUILD_TARGET "kineto_internal-build")
set(KINETO_DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(KINETO_SRC_PATH "${PROJECT_SOURCE_DIR}/third_party/kineto")
set(KINETO_BUILD_PATH "${KINETO_SRC_PATH}/build")
ExternalProject_Add(kineto_internal
                    PREFIX "third_party/kineto/"
                    DOWNLOAD_DIR ${KINETO_DOWNLOAD_PATH}
                    SOURCE_DIR ${KINETO_SRC_PATH}
                    SOURCE_SUBDIR libkineto
                    BINARY_DIR ${KINETO_BUILD_PATH}
                    DOWNLOAD_COMMAND ""
                    CMAKE_ARGS "-DKINETO_BUILD_TESTS=OFF"
                               "-DKINETO_USE_DEVICE_ACTIVITY=ON"
                               "-DKINETO_COMPILED_WITH_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI}"
                    BUILD_BYPRODUCTS "${KINETO_BUILD_PATH}/fmt/libfmt.a"
                                     "${KINETO_BUILD_PATH}/libkineto.a"
                    INSTALL_COMMAND cmake -E echo "Skipping install step for kineto_internal."
)
## The following code is a work around to avoid make file to run multiple externalProject-build when using make -j N
ExternalProject_Add_StepTargets(kineto_internal configure build)
ExternalProject_Add_StepDependencies(kineto_internal build kineto_internal-configure)
#-----------------------------------------------------------------------------------------------
add_library(kineto_lib STATIC IMPORTED GLOBAL)
add_library(kineto_fmt STATIC IMPORTED GLOBAL)
add_dependencies(kineto_lib kineto_internal-build)
add_dependencies(kineto_fmt kineto_internal-build)
set_target_properties(
  kineto_lib PROPERTIES IMPORTED_LOCATION
                        "${KINETO_BUILD_PATH}/libkineto.a")
set_target_properties(
  kineto_fmt PROPERTIES IMPORTED_LOCATION
                        "${KINETO_BUILD_PATH}/fmt/libfmt.a")
target_include_directories(kineto_fmt SYSTEM INTERFACE
  ${KINETO_SRC_PATH}/libkineto/third_party/fmt/include
)
target_include_directories(kineto_lib SYSTEM INTERFACE
  ${KINETO_SRC_PATH}/libkineto/include
  ${KINETO_SRC_PATH}/libkineto/src
  ${KINETO_SRC_PATH}/libkineto/third_party/fmt/include
)
add_library(kineto INTERFACE)
target_link_libraries(kineto INTERFACE kineto_fmt kineto_lib)
target_compile_definitions(kineto INTERFACE USE_KINETO)
#-----------------------------------------------------------------------------------------------

