# !/bin/bash
set -e
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

function run_coverage {
  if [ "$USE_COVERAGE" == "1" ]; then
    coverage run --source="$TORCH_DIPU_DIR" -p "$@"
  else
    python3 "$@"
  fi
}

function run_test {
  run_coverage "$@"
}

function run_op_tests {
  run_test "$CDIR/test_ops/test_adaptive_avg_pool2d_backward.py"
  run_test "$CDIR/test_ops/test_addmm.py"
  run_test "$CDIR/test_ops/test_log_softmax_backward.py"
  run_test "$CDIR/test_ops/test_log_softmax.py"
}

function run_tests {
  run_op_tests
#   run_dipu_tests
}

if [ "$LOGFILE" != "" ]; then
  run_tests 2>&1 | tee $LOGFILE
else
  run_tests
fi
