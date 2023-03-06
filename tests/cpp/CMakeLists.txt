
FILE(GLOB test1
testrt.cpp
)
# to use gtest
set(ALL_TESTS test_tensor_add test_relu testrt)
set(DIPU_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../torch_dipu/libtorch_dipu.so")
foreach (tname ${ALL_TESTS})
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} ${DIPU_LIB})
    target_link_libraries(${tname} c10 torch torch_cpu)
endforeach (tname)
