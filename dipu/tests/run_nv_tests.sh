# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
  unset DIPU_DUMP_OP_ARGS
  export PYTHONPATH=${DIPU_ROOT}/../:${PYTHONPATH}
  ${CDIR}/python/run_tests.sh
  echo "fill_.Scalar" >> .dipu_force_fallback_op_list.config
  run_test "${PYTORCH_DIR}/test/test_tensor_creation_ops.py" "$@" -v -f TestTensorCreationDIPU # --locals -f
  echo "" >  .dipu_force_fallback_op_list.config
  # run_test "${PYTORCH_DIR}/test/test_reductions.py" "$@" -v -f TestReductionsDIPU
  
  run_test "${PYTORCH_TEST_DIR}/test/nn/test_convolution.py" -v TestConvolutionNNDeviceTypeDIPU
  # run_test "${PYTORCH_TEST_DIR}/test/test_linalg.py" "$@" -v TestLinalgDIPU
  
  #　mock cuda cause test number err, temporary ignore
  # run_test "${PYTORCH_TEST_DIR}/test/test_testing.py" "$@" -v TestTestParametrizationDeviceTypeDIPU TestTestingDIPU
  run_test "${PYTORCH_TEST_DIR}/test/test_type_hints.py" "$@" -v
  run_test "${PYTORCH_TEST_DIR}/test/test_type_info.py" "$@" -v
  # run_test "${PYTORCH_TEST_DIR}/test/test_utils.py" "$@" -v
  run_test "${PYTORCH_TEST_DIR}/test/test_unary_ufuncs.py" "$@" -v TestUnaryUfuncsDIPU
  # run_test "${PYTORCH_TEST_DIR}/test/test_binary_ufuncs.py" "$@" -v TestBinaryUfuncsDIPU
  # run_test "${PYTORCH_TEST_DIR}/test/test_torch.py" "$@" -v TestTorchDeviceTypeDIPU #--subprocess
  run_test "${PYTORCH_TEST_DIR}/test/test_indexing.py" "$@" -v TestIndexingDIPU
  run_test "${PYTORCH_TEST_DIR}/test/test_indexing.py" "$@" -v NumpyTestsDIPU
  run_test "${PYTORCH_TEST_DIR}/test/test_view_ops.py" "$@" -v TestViewOpsDIPU
  # run_test "${PYTORCH_TEST_DIR}/test/test_type_promotion.py" "$@" -v TestTypePromotionDIPU
  # run_test "${PYTORCH_TEST_DIR}/test/test_nn.py" "$@" -v TestNN
  run_test "${PYTORCH_TEST_DIR}/test/test_ops_fwd_gradients.py" "$@" -v TestFwdGradientsDIPU
  run_test "${PYTORCH_TEST_DIR}/test/test_ops_gradients.py" "$@" -v TestBwdGradientsDIPU
  # run_test "${PYTORCH_TEST_DIR}/test/test_ops.py" "$@" -v
  run_test "${PYTORCH_TEST_DIR}/test/test_shape_ops.py" "$@" -v TestShapeOpsDIPU
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
