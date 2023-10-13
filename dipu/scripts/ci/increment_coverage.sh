# !/bin/bash
set -e
export LANG=en_US.UTF-8
ROOT_DIR=$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")
cd $ROOT_DIR && rm -rf coverage && mkdir coverage
echo "entering "$ROOT_DIR
require_coverage=$1

echo "==============C================"
lcov -c -d . --include "*/${ROOT_DIR#/mnt/*/}/torch_dipu/csrc_dipu/*" -o coverage/coverage.info
newcommit=`git rev-parse --short HEAD`
oldcommit=`git ls-remote origin main | cut -c 1-7`
if [ -z $oldcommit ];then echo "can not get main commit" && exit 1;fi
cd ${ROOT_DIR}/../ && git diff $oldcommit $newcommit --name-only | xargs -I {} realpath {} > dipu/coverage/gitdiff.txt 2>/dev/null || echo "error can be ignored"
cd $ROOT_DIR
for dir in `cat coverage/gitdiff.txt`;do
  skip=1
  buffer=""
  while IFS= read -r line; do
      if [[ $line == "TN:"* ]]; then
          buffer=$line
          skip=1
      elif [[ $line == *"SF:$dir" ]]; then
          skip=0
          echo "$buffer" >> "coverage/increment.info"
          echo "$line" >> "coverage/increment.info"
      elif [[ $skip -eq 0 ]]; then
          echo "$line" >> "coverage/increment.info"
      fi
      if [[ $line == "end_of_record" ]]; then
          skip=1
      fi
  done < "coverage/coverage.info"
done
echo "export IS_cover=True" > coverage/IS_cover.txt
if [ -f coverage/increment.info ];then
    lcov --list coverage/increment.info
    lcov --list coverage/increment.info > coverage/increment.txt
else
    echo "C无增量代码，或测试未覆盖到"
fi
python scripts/ci/increment_coverage.py $ROOT_DIR/coverage/ $require_coverage
source coverage/IS_cover.txt
if  [ $IS_cover == 'True' ];then
  exit 0
else
  echo "coverage does not exceed $require_coverage"
  exit 1
fi
