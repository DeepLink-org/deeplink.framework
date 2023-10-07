# !/bin/bash

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
export PYTHONPATH=$WORK_DIR:$PYTHONPATH
CONFIG_DIR=$(dirname $(readlink -f $0))
TEST_MODEL_DIR=${WORK_DIR}/model
LLAMA_MODEL_DIR="/home/cse/projects/pujiang/llama_infer/models"
export LLAMA_MODEL_DIR=$LLAMA_MODEL_DIR

BACKEND=ascendgraph
DYNAMIC=$1

PYTEST_CONFIG=${CONFIG_DIR}/${BACKEND}.ini
PYTEST_CONFIG_DYNAMIC=${CONFIG_DIR}/${BACKEND}_dynamic.ini
cd ${TEST_MODEL_DIR}
pytest -c ${PYTEST_CONFIG} --backend ${BACKEND} --dynamic false
if [ "$DYNAMIC" = "true" ]; then
    pytest -c ${PYTEST_CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic ${DYNAMIC}
fi
