set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

find_package(TOPSRT REQUIRED)
if(TOPSRT_FOUND)
    set(VENDOR_INCLUDE_DIRS ${TOPSRT_INCLUDE_DIR} ${TOPSRT_INCLUDE_DIR}/.. /usr/include/eccl/ PARENT_SCOPE)
    set(VENDOR_LIB_DIRS ${TOPSRT_LIBRARIES_DIR} PARENT_SCOPE)
    set(DIPU_VENDOR_LIB topsrt eccl PARENT_SCOPE)
    message("TOPSRT_INCLUDE_DIR:" ${TOPSRT_INCLUDE_DIR})
    message("TOPSRT_LIBRARIES_DIR:" ${TOPSRT_LIBRARIES_DIR})
    message("VENDOR_LIB_DIRS:" ${VENDOR_LIB_DIRS})
else()
    message(FATAL_ERROR "Not found TOPSRT.")
endif()

file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)
