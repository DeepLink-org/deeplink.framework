# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
    # TODO: Add PyTorch tests
    # run_test tests/test_ops/archived/test_tensor_add.py
  unset DIPU_DUMP_OP_ARGS
  export PYTHONPATH=${DIPU_ROOT}/../:${PYTHONPATH}

  ${CDIR}/python/run_tests.sh

  echo "fill_.Scalar" >> .dipu_force_fallback_op_list.config
  run_test "${PYTORCH_DIR}/test/test_tensor_creation_ops.py" "$@" -v -f TestTensorCreationDIPU # --locals -f
  echo "" >  .dipu_force_fallback_op_list.config

  #run_test "${PYTORCH_DIR}/test/test_linalg.py" "$@" -v -f TestLinalgDIPU
  echo "argmax.out,all.out,all.all_out,any.all_out,any.out" >> .dipu_force_fallback_op_list.config
  run_test "${PYTORCH_DIR}/test/test_reductions.py" "$@" -v -f TestReductionsDIPU

  run_test "${PYTORCH_DIR}/test/test_testing.py" "$@" -v -f TestTestParametrizationDeviceTypeDIPU TestTestingDIPU
  run_test "${PYTORCH_DIR}/test/test_type_hints.py" "$@" -v
  run_test "${PYTORCH_DIR}/test/test_type_info.py" "$@" -v
  #run_test "${PYTORCH_DIR}/test/test_utils.py" "$@" -v
  run_test "${PYTORCH_DIR}/test/test_unary_ufuncs.py" "$@" -v -f TestUnaryUfuncsDIPU
  run_test "${PYTORCH_DIR}/test/test_binary_ufuncs.py" "$@" -v -f TestBinaryUfuncsDIPU

  # need fix: random func test not throw expected err msg  as check_nondeterministic_alert() needed,
  # when device type is xpu it just ignore this err (should_alert= false), but device type 'cuda' will expose errors
  export DIPU_PYTHON_DEVICE_AS_CUDA=false
  run_test "${PYTORCH_DIR}/test/test_torch.py" "$@" -v -f TestTorchDeviceTypeDIPU #--subprocess
  export DIPU_PYTHON_DEVICE_AS_CUDA=true

  run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v -f TestIndexingDIPU
  run_test "${PYTORCH_DIR}/test/test_indexing.py" "$@" -v -f NumpyTestsDIPU
  run_test "${PYTORCH_DIR}/test/test_view_ops.py" "$@" -v -f TestViewOpsDIPU
  run_test "${PYTORCH_DIR}/test/test_type_promotion.py" "$@" -v -f TestTypePromotionDIPU
  #run_test "${PYTORCH_DIR}/test/test_nn.py" "$@" -v -f TestNN
  run_test "${PYTORCH_DIR}/test/test_ops_fwd_gradients.py" "$@" -v -f TestFwdGradientsDIPU
  run_test "${PYTORCH_DIR}/test/test_ops_gradients.py" "$@" -v -f TestBwdGradientsDIPU
  #run_test "${PYTORCH_DIR}/test/test_ops.py" "$@" -v
  run_test "${PYTORCH_DIR}/test/test_shape_ops.py" "$@" -v -f TestShapeOpsDIPU
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
