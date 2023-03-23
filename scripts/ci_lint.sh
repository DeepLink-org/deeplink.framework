# !/bin/bash
set -e

case $1 in
  py-lint)
    (echo "py-lint" && flake8 --ignore=E501,F841 python/conformance/diopi_functions.py \
       && flake8 --ignore=E501,F401 --exclude=python/conformance/diopi_functions.py,scripts/cpplint.py,impl/,python/conformance/model_config/ \
       && flake8 --ignore=E501,F401 python/conformance/model_config/process_config.py python/conformance/model_config/__init__.py ) \
    || exit -1;;
  cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    (echo "cpp-lint" && python scripts/cpplint.py --linelength=160 \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --recursive impl/ \
      && python scripts/cpplint.py --linelength=240 --filter=-build/header_guard --recursive diopirt/ ) \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;

esac
exit 0