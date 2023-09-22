project(test_ops)
cmake_minimum_required(VERSION 3.22)

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)

unset(PY_LIST)
list(APPEND PY_LIST "test_abs")
list(APPEND PY_LIST "test_add")
list(APPEND PY_LIST "test_exp")
list(APPEND PY_LIST "test_mm")


foreach(py_file IN LISTS PY_LIST)
  install(PROGRAMS ${py_file}.py
          DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

string(REGEX REPLACE "(.*)/(.*)/(.*)" "\\1/test_op" TEST_OP_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
include(test_ops.cmake)
