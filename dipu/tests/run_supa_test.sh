# !/bin/bash
set -ex

source tests/common.sh

function run_dipu_tests {
  # export DIPU_DUMP_OP_ARGS=2
  run_test "${PYTORCH_DIR}/test/test_tensor_creation_ops.py" "$@" -v -f TestTensorCreationDIPU # --locals -f

  # skip ne, embedding, pow, view_as_complex, view_as_real, transpose, softmax, gt,
  run_test "$CDIR/test_ops/archived/test_add.py"
  run_test "$CDIR/test_ops/archived/test_triu.py"
  run_test "$CDIR/test_ops/archived/test_triu.py"
  run_test "$CDIR/test_ops/archived/test_mean_std.py"
  run_test "$CDIR/test_ops/archived/test_rsqrt.py"
  run_test "$CDIR/test_ops/archived/test_mul.py"
  run_test "$CDIR/test_ops/archived/test_linear.py"
  run_test "$CDIR/test_ops/archived/test_matmul.py"
  run_test "$CDIR/test_ops/archived/test_transpose.py"
  run_test "$CDIR/test_ops/archived/test_div.py"
  run_test "$CDIR/test_ops/archived/test_silu.py"
  run_test "$CDIR/test_ops/archived/test_sort.py"
  run_test "$CDIR/test_ops/archived/test_cumsum.py"
  run_test "$CDIR/test_ops/archived/test_sub.py"
  run_test "$CDIR/test_ops/archived/test_multinomial.py"
  run_test "$CDIR/test_ops/archived/test_gather.py"
  run_test "$CDIR/test_ops/archived/test_where.py"
}

if [ "$LOGFILE" != "" ]; then
  run_dipu_tests 2>&1 | tee $LOGFILE
else
  run_dipu_tests
fi
