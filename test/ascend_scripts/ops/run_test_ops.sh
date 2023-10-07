# !/bin/bash

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
export PYTHONPATH=$WORK_DIR:$PYTHONPATH
CONFIG_DIR=$(dirname $(readlink -f $0))
TEST_OP_DIR=${WORK_DIR}/op

BACKEND=ascendgraph
NEED_DYNAMIC=$1

PYTEST_CONFIG=${CONFIG_DIR}/${BACKEND}.ini
PYTEST_CONFIG_DYNAMIC=${CONFIG_DIR}/${BACKEND}_dynamic.ini
cd ${TEST_OP_DIR}
if [ "$NEED_DYNAMIC" = "false" ]; then
    pytest -c ${PYTEST_CONFIG} --backend ${BACKEND} --need_dynamic ${NEED_DYNAMIC}
fi
pytest -c ${PYTEST_CONFIG_DYNAMIC} --backend ${BACKEND} --need_dynamic ${NEED_DYNAMIC}
