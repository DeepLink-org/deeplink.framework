# !/bin/bash

WORK_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$PYTHONPATH$WORK_DIR
TEST_MODEL=$1
BACKEND=$2
DYNAMIC=$3
LLAMA_MODEL_DIR="/home/cse/projects/pujiang/llama_infer/models"
export LLAMA_MODEL_DIR=$LLAMA_MODEL_DIR

cd ${WORK_DIR}
pytest -vs -rA test_${TEST_MODEL}.py --backend ${BACKEND} --dynamic ${DYNAMIC}
