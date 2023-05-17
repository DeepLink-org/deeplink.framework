# !/bin/bash
set -e

source tests/common.sh

function run_dipu_tests {
  # run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v TestTorchDeviceTypeDIPU
  # run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v TestIndexingDIPU
  run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v NumpyTestsDIPU
  run_test "${PYTORCH_DIR}/test/test_view_ops.py" "$@" -v TestViewOpsDIPU
  run_test "${PYTORCH_DIR}/test/test_view_ops.py" "$@" -v TestOldViewOpsDIPU
  run_test "${PYTORCH_DIR}/test/nn/test_pooling.py" "$@" -v TestPoolingNNDeviceTypeDIPU
  run_test "${PYTORCH_DIR}/test/nn/test_convolution.py" "$@" -v TestConvolutionNNDeviceTypeDIPU
  run_test "${PYTORCH_DIR}/test/test_unary_ufuncs.py" "$@" -v TestUnaryUfuncsDIPU
  run_test "${PYTORCH_DIR}/test/test_binary_ufuncs.py" "$@" -v TestBinaryUfuncsDIPU
  run_test "${PYTORCH_DIR}/test/test_linalg.py" "$@" -v TestLinalgDIPU
  run_test "${PYTORCH_DIR}/test/test_tensor_creation_ops.py" "$@" -v TestTensorCreationDIPU
  run_test "${PYTORCH_DIR}/test/test_tensor_creation_ops.py" "$@" -v TestRandomTensorCreationDIPU
  run_test "${PYTORCH_DIR}/test/test_tensor_creation_ops.py" "$@" -v TestLikeTensorCreationDIPU
  run_test "${PYTORCH_DIR}/test/test_tensor_creation_ops.py" "$@" -v TestAsArrayDIPU
  run_test "${PYTORCH_DIR}/test/test_nn.py" "$@" -v TestNNDeviceTypeDIPU
  run_test "${PYTORCH_DIR}/test/test_shape_ops.py" "$@" -v TestShapeOpsDIPU
  run_test "${PYTORCH_DIR}/test/test_reductions.py" "$@" -v TestReductionsDIPU
  run_test "${PYTORCH_DIR}/test/test_sparse.py" "$@" -v TestSparseUnaryUfuncsDIPU
  run_test "${PYTORCH_DIR}/test/test_sparse.py" "$@" -v TestSparseMaskedReductionsDIPU
  run_test "${PYTORCH_DIR}/test/test_sparse.py" "$@" -v TestSparseDIPU
  run_test "${PYTORCH_DIR}/test/test_sparse.py" "$@" -v TestSparseAnyDIPU
  run_test "${PYTORCH_DIR}/test/test_sort_and_select.py" "$@" -v TestSortAndSelectDIPU
  run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v TestVitalSignsCudaDIPU
  run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v TestTensorDeviceOpsDIPU
  run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v TestTorchDeviceTypeDIPU
  run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v TestDevicePrecisionDIPU
  run_test "${PYTORCH_DIR}/test/test_foreach.py" "$@" -v TestForeachDIPU
  # run_test "${PYTORCH_DIR}/test/test_type_promotion.py" "$@" -v TestTypePromotionDIPU
  # run_test "${PYTORCH_DIR}/test/test_nn.py" "$@"
  # run_test "${PYTORCH_DIR}/test/test_ops_fwd_gradients.py" "$@"
  # run_test "${PYTORCH_DIR}/test/test_ops_gradients.py" "$@"
  # run_test "${PYTORCH_DIR}/test/test_ops.py" "$@"
  # run_test "${PYTORCH_DIR}/test/test_shape_ops.py" "$@"
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
