#!/usr/bin/env bash

# TEST_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
if [ ! $TEST_DIR ]; then
    echo "TEST_DIR is not defined!" >&2
    exit 1
fi
TEST_OP_DIR=${TEST_DIR}/op
TEST_TORCH_OP=$1
BACKEND=$2
NEED_DYNAMIC=$3

cd ${TEST_OP_DIR}
pytest test_${TEST_TORCH_OP}.py --backend ${BACKEND} --need_dynamic ${NEED_DYNAMIC}
