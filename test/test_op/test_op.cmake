add_test (NAME py_${PY_FILE}
          COMMAND ${Python_EXECUTABLE} -m pytest ${TEST_OP_DIR}/${PY_FILE}.py -s -v -rA --backend ${BACKEND} --need_dynamic ${NEED_DYNAMIC})