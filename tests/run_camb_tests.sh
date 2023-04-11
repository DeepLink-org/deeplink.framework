# !/bin/bash
set -e

source tests/run_tests.sh
export DIPU_TEST_DEVICE="CAMB"

function run_dipu_tests {
  # run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v TestTorchDeviceTypeDIPU
  # run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v TestIndexingDIPU
  run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v NumpyTestsDIPU
  run_test "${PYTORCH_DIR}/test/test_view_ops.py" "$@" -v TestViewOpsDIPU
  # run_test "${PYTORCH_DIR}/test/test_type_promotion.py" "$@" -v TestTypePromotionDIPU
  run_test "$CDIR/test_ops/test_adaptive_avg_pool2d_backward.py"
  run_test "$CDIR/test_ops/test_addmm.py"
  run_test "$CDIR/test_ops/test_log_softmax_backward.py"
  run_test "$CDIR/test_ops/test_log_softmax.py"
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
