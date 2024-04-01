#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")" || exit 1

extract_first_version() {
  version_pattern="\b[0-9]+(\.[0-9]+)+\b"
  if [[ $1 =~ $version_pattern ]]; then
    echo "${BASH_REMATCH[0]}"
  else
    echo "Unknown"
  fi
}

get_cmd_version() {
  cmd=$1
  extract_first_version "$($cmd --version 2>&1 | head -n 1)"
}

check_cmd_version() {
  cmd=$1
  required_version=$2
  required_version_regex=^${required_version//x/[0-9]+}
  command -v "$cmd" >/dev/null || (echo "$cmd not found" && exit 1)
  current_version=$(get_cmd_version "$cmd")
  if [[ $current_version =~ $required_version_regex ]]; then
    echo "$cmd $required_version found, version: $current_version"
  else
    echo "WARNING! GitHub Actions CI uses $cmd $required_version, current version: $current_version"
  fi
}

# format all C/C++ files in current git repository with clang-format
check_cmd_version clang-format 17.x
CLANG_FORMAT_EXCLUDE_REGEX='^.*nlohmann/json\.hpp$'
git ls-files |\
  grep -E '^.+\.(c|h|cpp|cc|cxx|hpp|hh|hxx|cu|cuh)$' |\
  grep -Ev "$CLANG_FORMAT_EXCLUDE_REGEX" |\
  xargs clang-format -i --style=file

check_cmd_version black 24.x
# format all Python files in current git repository with black
# now only for dipu
git ls-files dipu |\
  grep -E '^.+\.py$' |\
  xargs black
