#!/bin/bash
set -euo pipefail

# Try finding clangd and libstdc++.so.6 on SCO.
# Note 1: ":+:" is used to handle unbound variable.
[ -f /mnt/cache/share/deeplinkci/github/proxy_on ] &&
    source /mnt/cache/share/deeplinkci/github/proxy_on
[ -d /mnt/cache/share/platform/dep/clang-17/bin ] &&
    export PATH=/mnt/cache/share/platform/dep/clang-17/bin${PATH:+:$PATH}

# Required tools.
[ -x "$(command -v git)" ] || { echo "::error::Missing git tool" && exit 1; }
[ -x "$(command -v clangd)" ] || { echo "::error::Missing clangd tool" && exit 1; }

# Get current folder.
self=$(dirname "$(realpath -s "${BASH_SOURCE[0]}")")
repo=$(cd "$self" && git rev-parse --show-toplevel)
tidy=${1:-$self/clangd-tidy}
tlog=${2:+-o\ $2}

# Download clangd-tidy scripts.
echo "check clangd-tidy"
[ -d "$tidy" ] ||
    git -c advice.detachedHead=false clone --depth 1 -b v0.2.0 https://github.com/lljbash/clangd-tidy.git "$tidy"

# Collect source files and run tidy.
echo "start clangd-tidy"
(cd "$repo/dipu" &&
    find torch_dipu ! -path '*/vendor/*' ! -name 'AutoGenedKernels.cpp' \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) |
    xargs "$tidy/clangd-tidy" --github --git-root="$repo" -j4 "$tlog")

echo "all done"
