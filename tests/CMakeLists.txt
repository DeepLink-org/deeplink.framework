set(ALL_TESTS test_tensor_add test_relu)

foreach (tname ${ALL_TESTS})
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} ${PLUGIN_NAME})
endforeach (tname)
