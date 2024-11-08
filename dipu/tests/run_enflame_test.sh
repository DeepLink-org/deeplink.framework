# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
  unset DIPU_DUMP_OP_ARGS
  export PYTHONPATH=${DIPU_ROOT}/../:${PYTHONPATH}
  ${CDIR}/python/run_tests.sh
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
