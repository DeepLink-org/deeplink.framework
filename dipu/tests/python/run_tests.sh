#!/usr/bin/env bash

function run_coverage {
  if [ "$USE_COVERAGE" == "ON" ]; then
    # We need to add "--concurrency=multiprocessing" to collect coverage data from
    # `multiprocessing.Process`. This flag requires all other configurations in
    # `.coveragerc`.
    cat << EOF > .coveragerc
[run]
  source = ${TORCH_DIPU_DIR}
  parallel = True
EOF
    coverage run --concurrency=multiprocessing "$@"
  else
    python "$@"
  fi
}

function run_test {
  run_coverage "$@"
}

cd $(dirname $0)
python ./individual_scripts/generate_unittest_for_individual_scripts.py \
    > ./unittests/unittest_autogened_for_individual_scripts.py
run_test -m unittest discover -s unittests \
    -p "*.py" \
    -v
