
FILE(GLOB test1
testrt.cpp
)

add_executable(cppm1 ${test1})

set(DIPU_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../torch_dipu/libtorch_dipu.so")
message(STATUS, "tests use dipu is:""${DIPU_LIB}")
 
target_link_libraries(cppm1 c10 torch torch_cpu)
target_link_libraries(cppm1 ${DIPU_LIB})
