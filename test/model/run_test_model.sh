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

TEST_MODEL_DIR=${TEST_DIR}/model
TEST_MODEL=$1
BACKEND=$2
DYNAMIC=$3

cd ${TEST_MODEL_DIR}
pytest -vs -rA test_${TEST_MODEL}.py --backend ${BACKEND} --dynamic ${DYNAMIC}
