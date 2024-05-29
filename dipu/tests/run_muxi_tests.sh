# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
  base_cuda_tests
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
