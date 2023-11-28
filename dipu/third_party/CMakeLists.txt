#require the external project to build as target
include(ExternalProject)

#[[ DIOPI ]]

# define some shared and cached options
set(WITH_DIOPI_LIBRARY "INTERNAL" CACHE STRING
  "Decide how to use DIOPI library, it could be one of (case-sensitive) \
  'INTERNAL', 'DISABLE' or '/directory/of/an/external/DIOPI/library'. \
  It's INTERNAL by default and uses the internal DIOPI library. \
  When it's DISABLE, DIPU won't link DIOPI. \
  When it's an absolute path, a user-provided DIOPI library will be used.")
set(WITH_DIOPI_INCLUDE "" CACHE PATH # use "PATH" type to provide a GUI file selector and convert relative path into absolute path
  "Provide a directory of external DIOPI header files, or use internal DIOPI if is empty. \
  When it's a directory, make sure WITH_DIOPI_LIBRARY is not INTERNAL.")

# locate DIOPI_LIBRARY_PATH
if(WITH_DIOPI_LIBRARY STREQUAL "INTERNAL")
  set(DIOPI_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/DIOPI/impl/lib")
  # the default path is hard-coded and not safe, better to use other methods
elseif(WITH_DIOPI_LIBRARY STREQUAL "DISABLE")
  set(DIOPI_LIBRARY_PATH "")
elseif(EXISTS "${WITH_DIOPI_LIBRARY}" AND IS_DIRECTORY "${WITH_DIOPI_LIBRARY}")
  set(DIOPI_LIBRARY_PATH "${WITH_DIOPI_LIBRARY}")
else()
  message(FATAL_ERROR
    "WITH_DIOPI_LIBRARY is invalid ('${WITH_DIOPI_LIBRARY}'), "
    "it should be one of 'INTERNAL', 'DISABLE' or an absolute path")
endif()

# locate DIOPI_INCLUDE_PATH
if (WITH_DIOPI_INCLUDE STREQUAL "")
  set(DIOPI_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/DIOPI/proto/include")
  # the default path is hard-coded and not safe, better to use other methods
elseif(EXISTS "${WITH_DIOPI_INCLUDE}" AND IS_DIRECTORY "${WITH_DIOPI_INCLUDE}" AND NOT WITH_DIOPI_LIBRARY STREQUAL "INTERNAL")
  set(DIOPI_INCLUDE_PATH "${WITH_DIOPI_INCLUDE}")
else()
  message(FATAL_ERROR
    "WITH_DIOPI_INCLUDE is invalid ('${WITH_DIOPI_INCLUDE}'). "
    "It should be empty or '/directory/of/DIOPI/headers' if WITH_DIOPI_LIBRARY is not INTERNAL.")
endif()

# compile DIOPI if use internal one
if (WITH_DIOPI_LIBRARY STREQUAL "INTERNAL")
  if(NOT DEFINED DIOPI_CMAKE_PREFIX_PATH)
    execute_process(
      COMMAND
        sh -x -c
        "python -c 'import torch;print(torch.utils.cmake_prefix_path)'"
      OUTPUT_VARIABLE DIOPI_CMAKE_PREFIX_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()

  message(STATUS "Building internal DIOPI with DIOPI_IMPL_OPT: ${DIOPI_IMPL_OPT}")
  ExternalProject_Add(diopi_internal
                      SOURCE_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/DIOPI"
                      SOURCE_SUBDIR "impl"
                      BINARY_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/DIOPI/build"
                      DOWNLOAD_COMMAND ""
                      CMAKE_ARGS
                        # note: as CMAKE_ARGS is a list, do not add quotes to arg values (such as "${DIOPI_IMPL_OPT}").
                        "-DIMPL_OPT=${DIOPI_IMPL_OPT}"
                        "-DENABLE_COVERAGE=${USE_COVERAGE}"
                        "-DCMAKE_PREFIX_PATH=${DIOPI_CMAKE_PREFIX_PATH}"
                        "-DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}"
                      BUILD_BYPRODUCTS "${DIOPI_LIBRARY_PATH}/libdiopi_impl.so"
                      INSTALL_COMMAND cmake -E echo "Skipping install step for diopi_internal.")

  ## The following code is a work around to avoid make file to run multiple externalProject-build when using make -j N
  ExternalProject_Add_StepTargets(diopi_internal configure build install)
  ExternalProject_Add_StepDependencies(diopi_internal install diopi_internal-build)
  ExternalProject_Add_StepDependencies(diopi_internal build diopi_internal-configure)
endif()

message(STATUS "Using DIOPI_LIBRARY_PATH='${DIOPI_LIBRARY_PATH}', DIOPI_INCLUDE_PATH='${DIOPI_INCLUDE_PATH}'")

add_library(diopi_impl INTERFACE)
target_include_directories(diopi_impl SYSTEM INTERFACE ${DIOPI_INCLUDE_PATH})
target_compile_definitions(diopi_impl INTERFACE DIOPI_ATTR_WEAK)

if(NOT WITH_DIOPI_LIBRARY STREQUAL "DISABLE")
  add_library(diopi_impl_lib SHARED IMPORTED)
  target_link_options(diopi_impl_lib INTERFACE "LINKER:-no-as-needed")
  set_target_properties(diopi_impl_lib PROPERTIES IMPORTED_LOCATION "${DIOPI_LIBRARY_PATH}/libdiopi_impl.so")

  add_dependencies(diopi_impl_lib diopi_internal-install)
  target_link_libraries(diopi_impl INTERFACE diopi_impl_lib)
endif()

#-------------------------add kineto as an external project ------------------------------------

include(FetchContent)
FetchContent_Declare(
  kineto
  URL           "${CMAKE_CURRENT_SOURCE_DIR}/kineto"
  SOURCE_SUBDIR libkineto
  SYSTEM)
set(KINETO_BUILD_TESTS OFF CACHE INTERNAL "turn off tests")
set(KINETO_USE_DEVICE_ACTIVITY ON CACHE INTERNAL "enable device activity")
set(KINETO_COMPILED_WITH_CXX11_ABI "${DIPU_COMPILED_WITH_CXX11_ABI}" CACHE INTERNAL "keep abi settings")

FetchContent_MakeAvailable(kineto)
