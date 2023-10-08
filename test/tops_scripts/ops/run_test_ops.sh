#!/usr/bin/env bash

# TEST_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
if [ ! $TEST_DIR ]; then
    echo "TEST_DIR is not defined!" >&2
    exit 1
fi
CONFIG_DIR=${TEST_DIR}/tops_scripts/ops
TEST_OP_DIR=${TEST_DIR}/op

BACKEND=topsgraph
NEED_DYNAMIC=$1

CONFIG_STATIC_ONLY=${CONFIG_DIR}/static_only.ini
CONFIG_STATIC_DYNAMIC=${CONFIG_DIR}/static_dynamic.ini
cd ${TEST_OP_DIR}
pytest -c ${CONFIG_STATIC_ONLY} --backend ${BACKEND} --need_dynamic false
pytest -c ${CONFIG_STATIC_DYNAMIC} --backend ${BACKEND} --need_dynamic ${NEED_DYNAMIC}
