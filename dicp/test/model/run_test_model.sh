#!/usr/bin/env bash

# export TOPS_VISIBLE_DEVICES=0
if [ ! $TEST_DIR ]; then
    echo "TEST_DIR is not defined!" >&2
    exit 1
fi


TEST_MODEL_DIR=${TEST_DIR}/model
TEST_MODEL=$1
BACKEND=$2
DYNAMIC=$3

if [ "$TEST_MODEL" == "llama" ] && [ ! $LLAMA_MODEL_DIR ]; then
    echo "LLAMA_MODEL_DIR is not defined!" >&2
    exit 1
fi
if [ "$TEST_MODEL" == "llama_finetune" ] && [ ! $LLAMA_FINETUNE_DIR ]; then
    echo "LLAMA_FINETUNE_DIR is not defined!" >&2
    exit 1
fi

cd ${TEST_MODEL_DIR}
if [ ${DYNAMIC} == false ] || [ ${DYNAMIC} == true ]; then
    pytest test_${TEST_MODEL}.py --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ $DYNAMIC == all ]; then
    pytest test_${TEST_MODEL}.py --backend ${BACKEND} --dynamic false
    pytest test_${TEST_MODEL}.py --backend ${BACKEND} --dynamic true
else
    echo "DYNAMIC should in (true, false, all)" >&2
    exit 1
fi
