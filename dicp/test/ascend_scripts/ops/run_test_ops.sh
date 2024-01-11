#!/usr/bin/env bash

# TEST_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
if [ ! $TEST_DIR ]; then
    echo "TEST_DIR is not defined!" >&2
    exit 1
fi
CONFIG_DIR=${TEST_DIR}/ascend_scripts/ops
TEST_OP_DIR=${TEST_DIR}/op

BACKEND=ascendgraph
DYNAMIC=$1

CONFIG_STATIC=${CONFIG_DIR}/static.ini
CONFIG_DYNAMIC=${CONFIG_DIR}/dynamic.ini

export TEST_DICP_INFER=1
cd ${TEST_OP_DIR}
if [ ${DYNAMIC} == false ]; then
    output=$(pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic ${DYNAMIC}  | tee /dev/tty )
elif [ ${DYNAMIC} == true ]; then
    output=$(pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic ${DYNAMIC}  | tee /dev/tty)
elif [ ${DYNAMIC} == all ]; then
    output=$(pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic false  | tee /dev/tty)
    output+="\n"
    output=$(pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic true  | tee /dev/tty)
else
    echo "DYNAMIC should in (true, false, all)" >&2
    unset TEST_DICP_INFER
    exit 1
fi
unset TEST_DICP_INFER

if echo "$output" | grep -q "FAILED"; then
    echo "ERROR! Not all pytest tests passed!" >&2
    exit 1
fi