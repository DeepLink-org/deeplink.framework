# !/bin/bash
set -exo pipefail

function builddipu() {
    path="build"
    echo "Building DIPU into: '$PWD/$path'"
    echo " - DIOPI_ROOT=${DIOPI_ROOT}"

    args=(
        "-DDEVICE=muxi"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "$@" )

    rm -rf "$path"
    mkdir -p "$path"
    cmake_maca -B "$path" -S . "${args[@]}" 2>&1 | tee "${path}/cmake_nv.log"
    cmake_maca --build "$path" --parallel 20 2>&1 | tee "${path}/build.log"
}


case $1 in
    "build_dipu")
        # build_diopi_lib
        builddipu  # "-DWITH_DIOPI_LIBRARY=${DIOPI_ROOT}"
    ;;
    "build_dipu_only")
        builddipu "-DWITH_DIOPI_LIBRARY=DISABLE" ;;
    *)
        echo "[ERROR] Incorrect option: $1" && exit 1 ;;
esac
