# to use gtest
set(ALL_TESTS test_tensor_add test_relu testrt)

foreach (tname ${ALL_TESTS})
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} torch_dipu)
endforeach (tname)
