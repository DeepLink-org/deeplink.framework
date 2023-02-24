set(NEUWARE_ROOT "/usr/local/neuware_2.4.1/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

# it's incorrect, need enhance find cmake to set lib fullpath
find_package(Neuware 4.6 REQUIRED)
set(VENDOR_INCLUDE_DIRS "${NEUWARE_ROOT_DIR}/include" PARENT_SCOPE)
set(VENDOR_LIB_DIRS "${NEUWARE_ROOT_DIR}/lib64" PARENT_SCOPE)


file(GLOB SRC_FILES  *.cpp)

set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)