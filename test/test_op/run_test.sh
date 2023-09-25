# !/bin/bash

WORK_DIR=$(cd $(dirname $0); pwd)
cmake -B build -S .
cmake build -DTEST_TORCH_OP=$1
cmake build -DBACKEND=$2
cmake build -DNEED_DYNAMIC=$3
cmake --build build --parallel
TIMESTAMP=$(date +%s)
LOG_FILE=${WORK_DIR}/build/tests_${TIMESTAMP}.log

ctest --test-dir build -V --rerun-failed --output-on-failure 2>&1 | tee ${LOG_FILE}