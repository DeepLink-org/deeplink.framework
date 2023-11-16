#!/usr/bin/env bash

# TEST_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
if [ ! $TEST_DIR ]; then
    echo "TEST_DIR is not defined!" >&2
    exit 1
fi
TEST_OP_DIR=${TEST_DIR}/op
TEST_TORCH_OP=$1
BACKEND=$2
DYNAMIC=$3

cd ${TEST_OP_DIR}
if [ ${DYNAMIC} == false ] || [ ${DYNAMIC} == true ]; then
    pytest test_${TEST_TORCH_OP}.py --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ $DYNAMIC == all ]; then
    pytest test_${TEST_TORCH_OP}.py --backend ${BACKEND} --dynamic false
    pytest test_${TEST_TORCH_OP}.py --backend ${BACKEND} --dynamic true
else
    echo "DYNAMIC should in (true, false, all)" >&2
    exit 1
fi
