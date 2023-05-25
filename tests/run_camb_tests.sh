# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
  run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v TestTorchDeviceTypeDIPU #--subprocess
  run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v TestIndexingDIPU
  run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v NumpyTestsDIPU
  run_test "${PYTORCH_DIR}/test/test_view_ops.py" "$@" -v TestViewOpsDIPU
  run_test "${PYTORCH_DIR}/test/test_type_promotion.py" "$@" -v TestTypePromotionDIPU
  #run_test "${PYTORCH_DIR}/test/test_nn.py" "$@" -v TestNN
  run_test "${PYTORCH_DIR}/test/test_ops_fwd_gradients.py" "$@" -v TestFwdGradientsDIPU
  run_test "${PYTORCH_DIR}/test/test_ops_gradients.py" "$@" -v TestBwdGradientsDIPU
  #run_test "${PYTORCH_DIR}/test/test_ops.py" "$@" -v
  run_test "${PYTORCH_DIR}/test/test_shape_ops.py" "$@" -v TestShapeOpsDIPU
  run_test "$CDIR/test_ops/test_adaptive_avg_pool2d_backward.py"
  run_test "$CDIR/test_ops/test_addmm.py"
  run_test "$CDIR/test_ops/test_log_softmax_backward.py"
  run_test "$CDIR/test_ops/test_log_softmax.py"
  ls $CDIR/test_ops/archived/test*.py | xargs --verbose  -I {} sh -c "python {}"
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
