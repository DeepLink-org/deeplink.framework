# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
  export DIPU_DUMP_OP_ARGS=1
  #run_test "${PYTORCH_DIR}/test/test_linalg.py" "$@" -v TestLinalgDIPU
  export DIPU_FORCE_FALLBACK_OPS_LIST=argmax.out
  run_test "${PYTORCH_DIR}/test/test_reductions.py" "$@" -v TestReductionsDIPU
  unset DIPU_FORCE_FALLBACK_OPS_LIST
  run_test "${PYTORCH_DIR}/test/test_testing.py" "$@" -v TestTestParametrizationDeviceTypeDIPU TestTestingDIPU
  run_test "${PYTORCH_DIR}/test/test_type_hints.py" "$@" -v
  run_test "${PYTORCH_DIR}/test/test_type_info.py" "$@" -v
  #run_test "${PYTORCH_DIR}/test/test_utils.py" "$@" -v
  run_test "${PYTORCH_DIR}/test/test_unary_ufuncs.py" "$@" -v TestUnaryUfuncsDIPU
  run_test "${PYTORCH_DIR}/test/test_binary_ufuncs.py" "$@" -v TestBinaryUfuncsDIPU
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
