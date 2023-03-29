# !/bin/bash
set -e

case $1 in
  py-lint)
    (
        echo "py-lint" && flake8 --ignore=E501,F841 torch_dipu/dipu
    ) \
    || exit -1;;
  cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    (echo "cpp-lint" && python scripts/cpplint.py --linelength=240 \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --recursive torch_dipu/) \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;

esac
exit 0