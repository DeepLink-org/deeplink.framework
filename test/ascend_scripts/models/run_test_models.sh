#!/usr/bin/env bash

# export TOPS_VISIBLE_DEVICES=0
# TEST_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
# LLAMA_MODEL_DIR=/home/cse/projects/pujiang/llama_infer/models
if [ ! $TEST_DIR ] || [ ! $LLAMA_MODEL_DIR ]; then
    if [ ! $TEST_DIR ]; then
        echo "TEST_DIR is not defined!" >&2
    fi
    if [ ! $LLAMA_MODEL_DIR ]; then
        echo "LLAMA_MODEL_DIR is not defined!" >&2
    fi
    exit 1
fi
CONFIG_DIR=${TEST_DIR}/ascend_scripts/models
TEST_MODEL_DIR=${TEST_DIR}/model

BACKEND=ascendgraph
DYNAMIC=$1

CONFIG_STATIC=${CONFIG_DIR}/static.ini
CONFIG_DYNAMIC=${CONFIG_DIR}/dynamic.ini
cd ${TEST_MODEL_DIR}

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