# !/bin/bash
set -exo pipefail

function build_dipu() {
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

# for reference only
function build_diopi_lib_dyn() {
    #  export PYTHONPATH="${YOU_TORCH_FOR_DIOPI}:${PYTHONPATH}"
    cd third_party/DIOPI/impl
    sh scripts/build_impl.sh clean
    sh scripts/build_impl.sh muxi_dyload || exit -1
    cd -
}

case $1 in
    "build_dipu")
        build_dipu
    ;;
    "build_diopi_dyn")
        build_diopi_lib_dyn
    ;;
    "build_dipu_only")
        # "-DWITH_DIOPI_LIBRARY=DISABLE"
        builddipu "-DWITH_DIOPI_LIBRARY=${DIOPI_DYN_ROOT}"
    ;;
    *)
        echo "[ERROR] Incorrect option: $1" && exit 1 ;;
esac
