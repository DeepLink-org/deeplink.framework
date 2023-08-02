# !/bin/bash
set -e
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

export TORCH_TEST_DEVICES="$CDIR/pytorch_test_base.py"
export USE_COVERAGE=ON
function run_coverage {
  if [ "$USE_COVERAGE" == "ON" ]; then
    coverage run --source="$TORCH_DIPU_DIR" -p "$@"
  else
    python "$@"
  fi
}

function run_test {
  run_coverage "$@"
}
