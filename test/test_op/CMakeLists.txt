project(test_op)
cmake_minimum_required(VERSION 3.22)

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)

set(PY_FILE test_${TEST_TORCH_OP})
install(PROGRAMS ${PY_FILE}.py
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_OP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(test_op.cmake)
