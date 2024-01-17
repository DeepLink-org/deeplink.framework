#!/bin/bash
set -euo pipefail
source /mnt/cache/share/deeplinkci/github/proxy_on
# Require Git.
[ -x "$(command -v git)" ] || (echo "missing git tool" && exit 1)

# Get current folder.
self=$(dirname $(realpath -s $0))
repo=$(cd $self && git rev-parse --show-toplevel)

# Download clangd-tidy scripts.
[ -d "$self/clangd-tidy" ] ||
    git -c advice.detachedHead=false clone --depth 1 -b v0.1.0 https://github.com/lljbash/clangd-tidy.git "$self/clangd-tidy"

# Try finding clangd and libstdc++.so.6 on 1988.
# Note: ":+:" is used to handle unbound variable.
[ -d /mnt/cache/share/platform/dep/clang-16/bin ] &&
    export PATH=/mnt/cache/share/platform/dep/clang-16/bin${PATH:+:$PATH}
[ -d /mnt/cache/share/platform/env/miniconda3.10/envs/pt2.0_diopi/lib ] &&
    export LD_LIBRARY_PATH=/mnt/cache/share/platform/env/miniconda3.10/envs/pt2.0_diopi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Forward srun commands.
# e.g. you can use: bash scripts/ci/nv/ci_nv_tidy.sh srun -p pat_rd
(cd "$repo/dipu" &&
    (find torch_dipu ! -path '*/vendor/*' ! -name AutoGenedKernels.cpp \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) |
        xargs $self/clangd-tidy/clangd-tidy -j4))
