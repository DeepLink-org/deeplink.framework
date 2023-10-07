# !/bin/bash

WORK_DIR=$(dirname $(readlink -f $0))
export PYTHONPATH=$PYTHONPATH$WORK_DIR
TEST_TORCH_OP=$1
BACKEND=$2
NEED_DYNAMIC=$3

cd ${WORK_DIR}
pytest test_${TEST_TORCH_OP}.py --backend ${BACKEND} --need_dynamic ${NEED_DYNAMIC}
