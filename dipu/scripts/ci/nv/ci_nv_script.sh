# !/bin/bash
set -eo pipefail

readonly OUTPUT="build"

function clean() {
    echo "Remove: '$PWD/$OUTPUT'"
    rm -rf "$OUTPUT"
}

function build() {
    echo "Building DIPU into: '$PWD/$OUTPUT'"

    args=(
        "-DDEVICE=cuda"
        "-DENABLE_COVERAGE=${USE_COVERAGE}"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "$@")

    mkdir -p "$OUTPUT"
    cmake -B "$OUTPUT" -S . "${args[@]}" 2>&1 | tee "${OUTPUT}/configure.log"
    cmake --build "$OUTPUT" --parallel 8 2>&1 | tee "${OUTPUT}/build.log"
}

# for reference only
function build_diopi_lib_dyn() {
    #  export PYTHONPATH="${YOU_TORCH_FOR_DIOPI}:${PYTHONPATH}"
    cd third_party/DIOPI/impl
    sh scripts/build_impl.sh clean
    sh scripts/build_impl.sh torch_dyload || exit -1
    cd -
}

case $1 in
"build_dipu")
    clean
    build
    ;;
"build_dipu_dev")
    build "${@:2}"
    ;;
"build_diopi_dyn")
    build_diopi_lib_dyn
    ;;
"build_dipu_only")
    clean
    build "-DWITH_DIOPI_LIBRARY=DISABLE"
    # build "-DWITH_DIOPI_LIBRARY=${DIOPI_DYN_ROOT}"
    ;;
*)
    echo "[ERROR] Incorrect option: $1" && exit 1
    ;;
esac
