set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

include(FindAscendToolKit)

# it's incorrect, need enhance find cmake to set lib fullpath
set(VENDOR_INCLUDE_DIRS "${ASCEND_TOOLKIT_ROOT}/include" PARENT_SCOPE)
set(VENDOR_LIB_DIRS "${ASCEND_TOOLKIT_ROOT}/lib64" PARENT_SCOPE)
set(DIPU_VENDOR_LIB ascendcl acl_op_compiler hccl PARENT_SCOPE)


# rewrite vendor header file path if needed
# set(UsedVendor "camb/XX" PARENT_SCOPE)

file(GLOB SRC_FILES  *.cpp  cnrt_${CNRT_VERSION}/*.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)