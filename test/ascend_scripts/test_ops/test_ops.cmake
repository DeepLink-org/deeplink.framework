add_test (NAME py_test_abs
          COMMAND ${Python_EXECUTABLE} -m pytest ${TEST_OP_DIR}/test_abs.py -s -v -rA --backend ascendgraph --need_dynamic True)

add_test (NAME py_test_add
          COMMAND ${Python_EXECUTABLE} -m pytest ${TEST_OP_DIR}/test_add.py -s -v -rA --backend ascendgraph --need_dynamic True)

add_test (NAME py_test_exp
          COMMAND ${Python_EXECUTABLE} -m pytest ${TEST_OP_DIR}/test_exp.py -s -v -rA --backend ascendgraph --need_dynamic True)

add_test (NAME py_test_mm
          COMMAND ${Python_EXECUTABLE} -m pytest ${TEST_OP_DIR}/test_mm.py -s -v -rA --backend ascendgraph --need_dynamic True)

