set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

# it's incorrect, need enhance find cmake to set lib fullpath
find_package(Neuware 4.6 REQUIRED)
set(VENDOR_INCLUDE_DIRS "${NEUWARE_ROOT_DIR}/include" PARENT_SCOPE)
set(VENDOR_LIB_DIRS "${NEUWARE_ROOT_DIR}/lib64" PARENT_SCOPE)
set(DIPU_VENDOR_LIB cnnl cncl cndev PARENT_SCOPE)

message(STATUS "Cambricon Neuware Root: ${NEUWARE_ROOT_DIR}")
message(STATUS "Cambricon CNRT Version: ${CNRT_VERSION}")
message(STATUS "Cambricon CNCC Flags: ${CMAKE_CNCC_FLAGS}")
message(STATUS "Cambricon CNCC Targets: ${CNCC_TARGETS}")


# rewrite vendor header file path if needed
# set(UsedVendor "camb/XX" PARENT_SCOPE)

file(GLOB SRC_FILES  *.cpp  cnrt_${CNRT_VERSION}/*.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)