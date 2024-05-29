# !/bin/bash
set -eox pipefail
export PS4='ln:${LINENO}: '

function build() {
    path="build"
    echo "Building DIPU into: '$PWD/$path'"
    echo " - DIOPI_ROOT=${DIOPI_ROOT}"

    args=(
        "-DDEVICE=muxi"
        "-DENABLE_COVERAGE=${USE_COVERAGE}"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "$@" )

    rm -rf "$path"
    mkdir -p "$path"
    cmake_maca -B "$path" -S . "${args[@]}" 2>&1 | tee "${path}/cmake_nv.log"
    cmake_maca --build "$path" --parallel 8 2>&1 | tee "${path}/build.log"
}

case $1 in
    "build_dipu")
        build "-DWITH_DIOPI_LIBRARY=${DIOPI_ROOT}" ;;
    "build_dipu_only")
        build "-DWITH_DIOPI_LIBRARY=DISABLE" ;;
    *)
        echo "[ERROR] Incorrect option: $1" && exit 1 ;;
esac
