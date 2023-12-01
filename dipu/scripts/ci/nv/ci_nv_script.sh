# !/bin/bash
set -e

function build() {
    path="build"
    echo "Building DIPU into: '$PWD/$path'"
    echo " - DIOPI_ROOT=${DIOPI_ROOT}"

    args=(
        "-DDEVICE=cuda"
        "-DENABLE_COVERAGE=${USE_COVERAGE}"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "$@" )

    rm -rf "$path"
    mkdir -p "$path"
    cmake -B "$path" -S . "${args[@]}" 2>&1 | tee "${path}/cmake_nv.log"
    cmake --build "$path" --parallel 8 2>&1 | tee "${path}/build.log"
}

case $1 in
    "build_dipu")
        (
            build
            python -c "import torch_dipu; print('build dipu successfully')"
        ) || exit -1;;
    "build_dipu_only")
        (
            build "-DWITH_DIOPI_LIBRARY=DISABLE"
            python -c "import torch_dipu; print('build dipu successfully')"
        ) || exit -1;;
    *)
        echo "[ERROR] Incorrect option: $1" && exit 1 ;;
esac
