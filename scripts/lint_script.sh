# !/bin/bash
set -e

current_path=$(cd "$(dirname "$0")"; pwd)
echo $current_path

case $1 in
  py-lint)
    (echo "py-lint" && flake8 --ignore=E501,F841,W503 dicp/) \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0
