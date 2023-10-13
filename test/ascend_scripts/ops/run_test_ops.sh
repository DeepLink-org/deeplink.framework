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

cd ${TEST_OP_DIR}
if [ ${DYNAMIC} == false ]; then
    pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ ${DYNAMIC} == true ]; then
    pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ ${DYNAMIC} == all ]; then
    pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic false
    pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic true
else
    echo "DYNAMIC should in (true, false, all)" >&2
    exit 1
fi
