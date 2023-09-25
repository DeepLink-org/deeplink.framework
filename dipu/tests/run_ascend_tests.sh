# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
    # TODO: Add PyTorch tests
    run_test tests/test_ops/archived/test_tensor_add.py
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
