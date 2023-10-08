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

CONFIG_STATIC_ONLY=${CONFIG_DIR}/static_only.ini
CONFIG_DYNAMIC_ONLY=${CONFIG_DIR}/dynamic_only.ini
cd ${TEST_MODEL_DIR}
pytest -c ${CONFIG_STATIC_ONLY} --backend ${BACKEND} --dynamic false
if [ "$DYNAMIC" = "true" ]; then
    pytest -c ${CONFIG_DYNAMIC_ONLY} --backend ${BACKEND} --dynamic ${DYNAMIC}
fi
