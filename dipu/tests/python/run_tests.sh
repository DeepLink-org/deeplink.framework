#!/usr/bin/env bash

cd $(dirname $0)
source ../common.sh
cd $(dirname $0)
python ./individual_scripts/generate_unittest_for_individual_scripts.py \
    > ./unittests/unittest_autogened_for_individual_scripts.py
run_test -m unittest discover -s unittests \
    -p "*.py" \
    -vf
