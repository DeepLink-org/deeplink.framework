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

add_library(diopi INTERFACE)
target_include_directories(diopi SYSTEM INTERFACE ${DIOPI_INCLUDE_PATH})
target_compile_definitions(diopi INTERFACE DIOPI_ATTR_WEAK)
target_link_options(diopi INTERFACE "LINKER:-no-as-needed")

if (WITH_DIOPI_LIBRARY STREQUAL "INTERNAL")
  # compile DIOPI if use internal one
  message(STATUS "Building internal DIOPI with DIOPI_IMPL_OPT: ${DIOPI_IMPL_OPT}")
  set(IMPL_OPT "${DIOPI_IMPL_OPT}" CACHE INTERNAL "IMPL_OPT for diopi")
  add_subdirectory(DIOPI/impl) # import diopi_impl target
  target_link_libraries(diopi INTERFACE diopi_impl)

elseif(NOT WITH_DIOPI_LIBRARY STREQUAL "DISABLE")
  add_library(diopi_impl_imported SHARED IMPORTED)
  set_target_properties(diopi_impl_imported PROPERTIES IMPORTED_LOCATION "${DIOPI_LIBRARY_PATH}/libdiopi_impl.so")
  target_link_libraries(diopi INTERFACE diopi_impl_imported)
endif()

message(STATUS "Using DIOPI_LIBRARY_PATH='${DIOPI_LIBRARY_PATH}', DIOPI_INCLUDE_PATH='${DIOPI_INCLUDE_PATH}'")

#[[ libkineto ]]

set(KINETO_BUILD_TESTS OFF CACHE INTERNAL "turn off tests")
set(KINETO_USE_DEVICE_ACTIVITY ON CACHE INTERNAL "enable device activity")
set(KINETO_COMPILED_WITH_CXX11_ABI "${DIPU_COMPILED_WITH_CXX11_ABI}" CACHE INTERNAL "pass through ABI settings")
# KINETO_COMPILED_WITH_CXX11_ABI might be removed from libkineto as we are
# using add_subdirectory instead of ExternalProject.
add_subdirectory(kineto/libkineto SYSTEM)
